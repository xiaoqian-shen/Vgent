import json
import re
import ast
import importlib
import numpy as np
import networkx as nx
from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer

from utils.prompts import *
from utils.retrieval import compute_text_similarity, extract_choices, allocate_node, node2indices, count_and_sort_filtered
from models.utils import resize_video

MODEL_MAP = {
    "llava_video":    ("models.llavavideo", "lmms-lab/LLaVA-Video-7B-Qwen2"),
    "qwenvl25_7b":    ("models.qwenvl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwenvl25_3b":    ("models.qwenvl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwenvl2_7b":     ("models.qwenvl", "Qwen/Qwen2-VL-7B-Instruct"),
    "qwenvl2_2b":     ("models.qwenvl", "Qwen/Qwen2-VL-2B-Instruct"),
    "internvl25_2b":  ("models.internvl", "OpenGVLab/InternVL2_5-2B"),
    "longvu":         ("models.longvu", "Vision-CAIR/LongVU_Qwen2_7B"),
}

class Vgent():
    def __init__(self, args):
        self.args = args
        module_name, model_path = next(
            ((module, model_path) for key, (module, model_path) in MODEL_MAP.items() if key in self.args.model_name),
            None
        )

        module = importlib.import_module(module_name)
        self.mllm_response, self.load_video, self.load_model = (
            module.mllm_response,
            module.load_video,
            module.load_model,
        )
        self.processor, self.video_llm, self.image_processor, _ = self.load_model(model_path)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
    
    def generate_entities(self, prompt, video_input, max_new_tokens=512):
        attempts = 0
        while attempts < 5:
            try:
                response = self.mllm_response(self.video_llm, self.processor, self.image_processor, prompt, None, video_input, max_new_tokens)
                info = json.loads(response.replace("```json", "").replace("```","").strip())
                
                entities = [f"{entity['entity name']}, {entity['description']}" 
                            for entity in info.get("entities", []) 
                            if "entity name" in entity and "description" in entity]
                
                actions = [f"{entity['entity name']}, {entity['action description']}" 
                        for entity in info.get("actions", []) 
                        if "entity name" in entity and "action description" in entity]
                
                scenes = [scene["location"] for scene in info.get("scenes", []) if "location" in scene]
                
                return entities, actions, scenes
            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                attempts += 1
        
        return [], [], []
    
    def construct_graph(self, video_inputs, subtitles):
        split_video_inputs = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
        video_graph = nx.DiGraph()
        entity_graph = defaultdict(set)
        for idx, video_input in enumerate(split_video_inputs):
            entities, actions, scenes = self.generate_entities(GRAPH_PROMPT, video_input, max_new_tokens=512)
            if subtitles is not None:
                start_time = idx * self.args.chunk_size // self.args.fps
                end_time = (idx + 1) * self.args.chunk_size // self.args.fps
                current_subtitles = [text for time, text in subtitles if time >= start_time and time < end_time]
            else:
                current_subtitles = None
            video_graph.add_node(idx, actions=actions, scenes=scenes, entities=entities, subtitles=current_subtitles)
            for entity in entities + actions + scenes:
                entity_name = entity.split(',')[0].lower()
                if len(entity_graph) == 0:
                    entity_graph[entity_name].add(idx)
                    continue
                entity_sim = compute_text_similarity([entity], list(entity_graph.keys()), self.embedding_model, self.embedding_tokenizer, return_all=True)
                max_sim_idx = max(range(len(entity_sim)), key=lambda i: entity_sim[i])
                max_sim = entity_sim[0][max_sim_idx]
                if max_sim > 0.7:
                    most_similar_entity = list(entity_graph.keys())[max_sim_idx]
                    entity_graph[most_similar_entity].add(idx)
                    video_graph.add_edges_from((idx, i, {"label": most_similar_entity}) for i in entity_graph[most_similar_entity])

                else:
                    entity_graph[entity_name].add(idx)
        return video_graph, entity_graph
    
    def extract_keywords(self, question, candidates, video_inputs):
        reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
        flag = True
        count = 0
        llm_info = None
        while flag and count < 5:
            try:
                response = self.mllm_response(self.video_llm, self.processor, self.image_processor, reason_prompt, None, None, max_new_tokens=256)
                llm_info = json.loads(response.replace("```json", "").replace("```","").strip())
                flag = False
            except:
                count += 1
                continue
        
        query_list = llm_info["keywords"] if llm_info is not None and "keywords" in llm_info else []
        query_list = query_list + candidates
        query_list = list(set(query_list))

        return query_list, llm_info
    
    def retrieve_nodes(self, question, query_list, video_inputs, candidates, video_graph, entity_graph, subtitles, llm_info):
        indices = None
        if "subtitle" in question.lower() and subtitles is not None and re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question):
            query_subtitle = re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
            indices = []
            for time, text in subtitles:
                if text in query_subtitle:
                    indices.append(time)
            node_list = []
        elif 'beginning' in question.lower() or 'at the start of' in question.lower():
            node_list = [i for i in range(3)]
        elif 'at the end of the video' in question.lower():
            node_list = [i for i in range(max(round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)) - 3, 0), round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)))]
        elif video_graph is None:
            node_list = list(range(round(np.ceil(len(video_inputs[0]) / self.args.chunk_size)))) if (llm_info is not None and "tool" in llm_info and llm_info["tool"] in ["action counting", "order"] and self.args.task == 'mlvu') or len(video_inputs[0]) <= 128 else []
        else:
            if "order" in question.lower():
                query_list = extract_choices(question, candidates)
            query_list.append(question)
            node_list = allocate_node(self.args, video_graph, entity_graph, query_list, self.embedding_model, self.embedding_tokenizer)
            key_list = []
            for node_id in node_list:
                node_data = video_graph.nodes[node_id]
                if node_data.get('subtitles') is None:
                    key_list.append("; ".join(node_data.get('entities', '')) + "; ".join(node_data.get('actions', '')) + "; ".join(node_data.get('scenes', '')))
                else:
                    key_list.append("; ".join(node_data.get('entities', '')) + "; ".join(node_data.get('actions', '')) + "; ".join(node_data.get('scenes', '')) + "; ".join(node_data.get('subtitles', '')))
            sims = compute_text_similarity(query_list, key_list, self.embedding_model, self.embedding_tokenizer, return_all=True)
            sorted_indices = torch.argsort(torch.mean(sims, dim=0), descending=True)
            node_list = [node_list[i] for i in sorted_indices]
        return {"nodes": node_list[:self.args.n_retrieval], "indices": indices}

    def refine_nodes(self, retrieved_node_list, question, llm_info, candidates, video_inputs, subtitles, size_list=None):
        if len(retrieved_node_list["nodes"]) == 0:
            return retrieved_node_list, None, None
        input_candidates = " ".join(candidates)
        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None
        prompt = SQL_PROMPT.format(query=question, candidates=input_candidates)
        info = None
        count = 0
        obj_count = False
        if question_type == "order" or "order" in question.lower():
            choices = extract_choices(question, candidates)
            info = {}
            for choice in choices:
                info[f"Q{choices.index(choice) + 1}"] = f"Is '{choice.lower()}' shown in video?"
        elif question_type == "action counting" and 'action' in question.lower():
            try:
                match = re.search(r"'(.*?)'", question)
                extracted_text = match.group(1)
                info = {"Q1": f"Is there a scene featuring the '{extracted_text}' action in the video?"}
            except:
                info = None
        elif question_type == "object counting" or 'how many' in question.lower():
            obj_count = True
            info = {"Q1": question}
        flag = True
        if info is None:
            while flag and count < 5:
                try:
                    response = self.mllm_response(self.video_llm, self.processor, self.image_processor, prompt, None, None, 512)
                    info.update(json.loads(response.replace("```json", "").replace("```","").strip()))
                    flag = False
                except:
                    count += 1
                    continue
        if info is None:
            return retrieved_node_list, None, None
        split_video_inputs = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
        split_size_list = torch.split(size_list, self.args.chunk_size, dim=0) if size_list is not None else None
        check_result = {}
        for node in retrieved_node_list["nodes"]:
            if node >= len(split_video_inputs):
                continue
            video_input = split_video_inputs[node]
            size_list_input = split_size_list[node] if split_size_list is not None else None
            if subtitles is not None:
                subtitle_prompt = " This video's subtitles are listed below:\n"
                start_time = node * self.args.chunk_size // self.args.fps
                end_time = (node + 1) * self.args.chunk_size // self.args.fps
                select_subtitles = [text for time, text in subtitles if time >= start_time and time < end_time]
                subtitle_prompt += " ".join(select_subtitles) + "\n"
            else:
                subtitle_prompt = ""
            input_candidates = " ".join(candidates)
            instruct = SQL_ANSWER_COUNT_PROMPT.format(questions=info) + subtitle_prompt if obj_count else SQL_ANSWER_PROMPT.format(questions=info) + subtitle_prompt
            try:
                output_text = self.mllm_response(self.video_llm, self.processor, self.image_processor, instruct, None, video_input, max_new_tokens=256, size_list=size_list_input)
                pred = json.loads(output_text.replace("```json", "").replace("```","").strip())
            except:
                pred = None
            check_result[node] = pred
        count_dict, sorted_nodes = count_and_sort_filtered(check_result)
        retrieved_node_list["nodes"] = sorted_nodes
        return retrieved_node_list, info, check_result
    
    def aggregate_nodes(self, refined_node_list, llm_info, video_inputs, raw_video, size_list, subtitles, prompt, query, video_graph, sql_check, check_result, fps):
        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None
        select_subtitles = None
        node_list = refined_node_list["nodes"]
        if node_list is not None and len(node_list) > 0:
            indices, sorted_node_list = node2indices(node_list, question_type, video_inputs, self.args)
            video_segments = video_inputs[0][indices]
            input_size_list = size_list[indices] if size_list is not None else None
            if subtitles is not None:
                if video_graph is None:
                    select_subtitles = []
                    for node_id in sorted_node_list:
                        select_subtitles.extend([text for time, text in subtitles if time >= node_id * self.args.chunk_size and time < (node_id + 1) * self.args.chunk_size])
                else:
                    select_subtitles = [text for time, text in subtitles]
        elif refined_node_list["indices"] is not None:
            indices = refined_node_list["indices"]
            input_size_list = size_list[indices] if size_list is not None else None
            extend_indices = []
            if subtitles is not None:
                select_subtitles = []
                for index in indices:
                    select_subtitles.extend([text for time, text in subtitles if time == index])
                    extend_indices.extend(list(range(max(0, index - 10), min(len(video_inputs[0]) - 1, index + 10))))
            indices = sorted(set(extend_indices))
            video_segments = video_inputs[0][indices]
        else:
            indices = np.linspace(0, len(video_inputs[0]) - 1, min(self.args.uniform_frame, len(video_inputs[0])), dtype=int)
            video_segments = video_inputs[0][indices]
            input_size_list = size_list[indices] if size_list is not None else None
            if subtitles is not None:
                select_subtitles = [text for time, text in subtitles]
        
        if select_subtitles is not None:
            subtitle_prompt = "This video's subtitles are listed below:\n"
            subtitle_prompt += " ".join(select_subtitles) + "\n"
            input_prompt = subtitle_prompt + prompt
        else:
            input_prompt = prompt
        
        if question_type == "action counting" and node_list is not None and (None not in [re.search(r'\d+', c) for c in query['candidates']]):
            numbers = [int(re.search(r'\d+', c).group()) for c in query['candidates']]
            pred_idx = min(range(len(numbers)), key=lambda i: abs(numbers[i] - len(node_list)))
            pred = query['letters'][pred_idx]
        else:
            agg_info = None
            multiple = llm_info["multiple"] if llm_info is not None and "multiple" in llm_info else 'no'
            if sql_check is not None and check_result is not None and node_list is not None and len(node_list) > 1 and multiple == "yes" and question_type != "object counting":
                input_text = ""
                for key, value in check_result.items():
                    input_text += f"video [{key}]:\n"
                    for question_id, question in sql_check.items():
                        if key in check_result and check_result[key] is not None and question_id in check_result[key] and check_result[key][question_id] != 'no':
                            input_text += f"{question}: {check_result[key][question_id]}\n"
                input_candidates = " ".join(query['candidates'])
                input_text = AGGREGATE_PROMPT.format(query=query['question'], candidates=input_candidates, input=input_text)
                try:
                    agg_info = self.mllm_response(self.video_llm, self.processor, self.image_processor, input_text, None, None, 128)
                except:
                    agg_info = None
            input_prompt = input_prompt + PRED_PROMPT + agg_info if agg_info is not None else input_prompt + PRED_PROMPT
            if "qwenvl" in self.args.model_name:
                # qwenvl support dynamic resolution
                video_segments = raw_video[0][indices]
                if len(video_segments) == 0:
                    video_segments = raw_video[0]
                video_segments, fps = resize_video(video_segments, fps, total_pixels=self.args.total_pixels*28*28, maximum_frames=512)
            output_text = self.mllm_response(self.video_llm, self.processor, self.image_processor, input_prompt, None, video_segments, max_new_tokens=10, size_list=input_size_list, fps=fps)
            pred_answer = output_text.strip("()").strip()
            if pred_answer in query['letters']:
                pred_idx = query['letters'].index(pred_answer)
            else:
                # unmatched letter
                pred_idx = 2
            pred = query['letters'][pred_idx]

        return pred