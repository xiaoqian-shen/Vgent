import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--model_name', default="qwenvl25_7b")
    parser.add_argument('--output_path', default="./eval")
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--task', default="mlvu")
    parser.add_argument('--data_path', default="./data")
    parser.add_argument('--uniform_frame', type=int, default=450)
    parser.add_argument('--n_retrieval', type=int, default=20)
    parser.add_argument('--n_refine', type=int, default=5)
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--graph_path', type=str, default='./graphs')
    parser.add_argument('--total_pixels', type=int, default=16384)
    parser.add_argument('--duration', default="long,medium,short", type=str)

    args = parser.parse_args()
    args.duration = args.duration.split(",")

    return args
