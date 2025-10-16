import math
import torch
import os
import json
from transformers.trainer_pt_utils import IterableDatasetShard
import datetime
from tqdm import tqdm
import pysubs2
import argparse
from utils.data import EvalDatasetMLVU, EvalDatasetVideoMME, EvalDatasetLongVideoBench, get_subtitles
from utils.vgent import Vgent
import pickle
from torch import distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', default=0)
parser.add_argument('--model_name', default="qwenvl25_7b")
parser.add_argument('--chunk_size', type=int, default=64)
parser.add_argument('--task', default="mlvu")
parser.add_argument('--n_retrieval', type=int, default=20)
parser.add_argument('--n_refine', type=int, default=5)
parser.add_argument('--data_path', default="./data")
parser.add_argument('--fps', type=float, default=1.0)
parser.add_argument('--graph_path', type=str, default='./graphs')
parser.add_argument('--total_pixels', type=int, default=16384)
args = parser.parse_args()

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
torch.distributed.barrier()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
world_size = torch.distributed.get_world_size()
world_rank = torch.distributed.get_rank()

vgent = Vgent(args)

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
    video_path = line.get("video_path", None)
    subtitle_path = line.get("subtitle", None)
        
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

    if os.path.exists(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}/{video_name.split('.')[0]}.pkl") or len(video_inputs[0]) < args.chunk_size * args.n_retrieval:
        continue
    else:
        video_graph, entity_graph = vgent.construct_graph(video_inputs, subtitles)
        os.makedirs(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}", exist_ok=True)
        pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph}, open(f"{args.graph_path}/{args.task}_{args.fps}fps_{args.chunk_size}/{video_name.split('.')[0]}.pkl", 'wb'))
