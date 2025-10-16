import torch
import numpy as np
import json
import re
from utils.prompts import SQL_PROMPT, SQL_ANSWER_PROMPT, PRED_PROMPT, SQL_ANSWER_COUNT_PROMPT, REASONING_PROMPT
from models.utils import resize_video

def compute_text_similarity(query_list, key_list, embedding_model, tokenizer, return_all=False):
    encoded_input = tokenizer(query_list + key_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
        embeddings = model_output[0][:, 0]
    query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
    key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
    sims = query_emb @ key_emb.T
    if return_all:
        return sims
    else:
        return torch.mean(sims)

def node2indices(node_list, question_type, video_inputs, args):
    n_refine = 8 if question_type == "order" else args.n_refine
    sorted_node_list = sorted(map(int, node_list[:n_refine]))
    indices = []
    for idx in sorted_node_list:
        start_time = idx * args.chunk_size
        end_time = min((idx + 1) * args.chunk_size, len(video_inputs[0]))
        indices.extend(range(start_time, end_time))
    indices = set(indices)
    indices = sorted(indices)
    return indices, sorted_node_list

def allocate_node(args, video_graph, entity_graph, query_list, embedding_model, tokenizer, threshold=0.5):
    node_list = []
    for key in list(entity_graph.keys()):
        if compute_text_similarity(query_list, [key], embedding_model, tokenizer) > threshold:
            node_list.extend(entity_graph[key])
    for (node, data) in video_graph.nodes(data=True):
        if node in node_list:
            continue
        if data.get('subtitles') is None:
            key_list = data.get('entities', []) + data.get('actions', []) + data.get('scenes', []) + data.get('scenes', [])
        else:
            key_list = data.get('entities', []) + data.get('actions', []) + data.get('scenes', []) + data.get('scenes', []) + data.get('subtitles', [])
        if compute_text_similarity(query_list, key_list, embedding_model, tokenizer) > threshold:
            node_list.append(node)

    node_list = list(set(node_list))
    return node_list

def extract_choices(question, candidates):
    if "(1)" in question or "(a)" in question:
        pattern = r'\(([a-zA-Z0-9]+)\)\s*(.+?)(?=\s*\([a-zA-Z0-9]+\)|$)'
        matches = re.findall(pattern, question, flags=re.DOTALL)
        query_list = [c[1].strip() for c in matches]
    elif re.search(r"\d+\.", question):
        pattern = r"\d+\.\s+([^\n]+)"
        matches = re.findall(pattern, question)
        query_list = [match.strip() for match in matches]
    elif "-->" in candidates[0]:
        choices = candidates[0].split("-->")
        query_list = [choice.strip() for choice in choices]
    elif len(candidates[0].split(",")) > 2:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    elif len(candidates[0].split(",")) > 1 and 'and' in candidates[0]:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            choices = choices.replace(' and ', ',')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    else:
        query_list = candidates
    return query_list

def count_and_sort_filtered(data):
    count_dict = {}
    for index, answers in data.items():
        if answers is not None and isinstance(answers, dict):
            for key, value in answers.items():
                if value != 'no' and value != '0' and value != 0:
                    count_dict[index] = count_dict.get(index, 0) + 1

    filtered_dict = {k: v for k, v in count_dict.items() if v > 0}

    sorted_indices = sorted(filtered_dict.keys(), key=lambda k: filtered_dict[k], reverse=True)

    return filtered_dict, sorted_indices

