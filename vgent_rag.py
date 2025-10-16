import math
import torch
import os
import json
from transformers.trainer_pt_utils import IterableDatasetShard
import datetime
from tqdm import tqdm
import pysubs2
import argparse
from itertools import chain
from utils.data import EvalDatasetMLVU, EvalDatasetVideoMME, EvalDatasetLongVideoBench, get_subtitles
from utils.vgent import Vgent
import pickle
from torch import distributed as dist
from utils.config import get_args

args = get_args()
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
torch.distributed.barrier()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
world_size = torch.distributed.get_world_size()
world_rank = torch.distributed.get_rank()
checkpoint_dir = os.path.join(f"{args.output_path}/{args.model_name}/{args.task}")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{world_rank}.json")

vgent = Vgent(args)

processed_identifiers = set() 
output = []
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
        output = checkpoint_data.get('output', [])
        for item in output:
            if 'video_name' in item and 'question' in item:
                identifier = (item['video_name'], item['question']) 
                processed_identifiers.add(identifier)
    print(f"Rank {world_rank}: Resuming with {len(output)} already processed question-video pairs.")

if args.task == "mlvu":
    dataset = EvalDatasetMLVU(data_path=args.data_path)
elif args.task == "videomme":
    dataset = EvalDatasetVideoMME(data_path=args.data_path)
elif args.task == "lvb":
    dataset = EvalDatasetLongVideoBench(data_path=args.data_path)

shard_dataset = IterableDatasetShard(
    dataset,
    batch_size=1,
    num_processes=world_size,
    process_index=world_rank,
)

torch.distributed.barrier()
total_videos_for_rank = len(list(shard_dataset))
pbar = tqdm(shard_dataset, total=total_videos_for_rank, desc=f"Rank {world_rank} Processing Videos")

for line in pbar:
    video_name = line.get("video_name", None)
    answer = line.get("answer", None)
    prompt = line.get("prompt", None)
    question = line.get("question", None)
    task_type = line.get("task_type", None)
    video_path = line.get("video_path", None)
    candidates = line.get("candidates", None)
    subtitle_path = line.get("subtitle", None)
    duration = line.get("duration", None)

    current_identifier = (video_name, question)
    if current_identifier in processed_identifiers:
        continue

    if not os.path.exists(video_path):
        print(video_path)
        continue
    try:
        raw_video, _, _, frame_idx, fps, video_inputs, size_list = vgent.load_video(video_path, args)
        if "llava_video" in args.model_name:
            video = vgent.image_processor.preprocess(raw_video, return_tensors="pt")["pixel_values"].cuda().to(dtype=torch.bfloat16)
            video_inputs = [video]
        if type(video_inputs) is not list:
            video_inputs = [video_inputs]
    except:
        continue

    subtitles = get_subtitles(subtitle_path, len(video_inputs[0]), fps=args.fps, data=line)

    if len(video_inputs[0]) < args.chunk_size * args.n_retrieval:
        video_graph, entity_graph = (None, None)
    elif os.path.exists(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}/{video_name.split('.')[0]}.pkl"):
        saved_graph = pickle.load(open(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}/{video_name.split('.')[0]}.pkl", "rb"))
        video_graph, entity_graph = (saved_graph["video_graph"], saved_graph["entity_graph"]) if len(video_inputs[0]) > args.chunk_size * args.n_retrieval else (None, None)
    else:
        video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles)
        os.makedirs(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}", exist_ok=True)
        pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph}, open(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}/{video_name.split('.')[0]}.pkl", 'wb'))
    
    query_list, llm_info = vgent.extract_keywords(question, candidates, video_inputs)
    retrieved_node_list = vgent.retrieve_nodes(question, query_list, video_inputs, candidates, video_graph, entity_graph, subtitles, llm_info)
    refined_node_list, sql_check, check_result = vgent.refine_nodes(retrieved_node_list, question, llm_info, candidates, video_inputs, subtitles, size_list)
    pred = vgent.aggregate_nodes(refined_node_list, llm_info, video_inputs, raw_video, size_list, subtitles, prompt, line, video_graph, sql_check, check_result, fps)

    output.append(
        {
            "question": question,
            "candidates": candidates,
            "task_type": duration if args.task == "videomme" else task_type,
            "video_name": video_name,
            "duration": len(video_inputs[0]),
            "domain": line.get("domain", None),
            "sub_category": line.get("sub_category", None),
            "video_id": line.get("video_id", None),
            "node_list": refined_node_list["nodes"][:args.n_refine],
            "info": llm_info,
            "sql_check": sql_check,
            "check_result": check_result,
            "pred": pred,
            "answer": answer,
        }
    )
    print(output[-1], flush=True)

    processed_identifiers.add(current_identifier)
    with open(checkpoint_file, 'w') as f:
        json.dump({'output': output, 'processed_identifiers': list(processed_identifiers)}, f)
    
    print(f"Rank {world_rank} Output for {video_name[:8]}... - {question[:20]}...: {output[-1]['pred']}, answer: {answer}", flush=True)
    
dist.barrier()
    
final_output = [None] * world_size
dist.all_gather_object(
    final_output,
    output,
)
all_output = list(chain(*final_output))

global_rank = dist.get_rank()
if global_rank == 0:
    output_filename = os.path.join(checkpoint_dir, f"output.json")
    with open(output_filename, "w") as f:
        json.dump(all_output, f)
    
    result = {}
    task_types = set([item['task_type'] for item in all_output])
    for task_type in task_types:
        task_type_output = [item for item in all_output if item['task_type'] == task_type]
        accuracy = sum(1 for item in task_type_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(task_type_output)
        result[task_type] = accuracy
    result["overall"] = sum(1 for item in all_output if item['answer'] in item['pred'] or item['pred'] in item['answer']) / len(all_output)
    print(result)
    
    result_filename = os.path.join(checkpoint_dir, f"result.json")
    with open(result_filename, "w") as f:
        json.dump(result, f)

    for rank_idx in range(world_size):
        rank_checkpoint_file = os.path.join(checkpoint_dir, f"cuda:{rank_idx}.json")
        if os.path.exists(rank_checkpoint_file):
            os.remove(rank_checkpoint_file)
