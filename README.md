# Vgent

<div align="center">
    <h4>
        <div>
          Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding
        </div>
    </h4>
</div>

<div align="center">
    <a href="" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoWidth=15" height="20" /></a>
    <a href="https://xiaoqian-shen.github.io/Vgent" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Webpage-Vgent-blue?style=flat-square" height="20" /></a>
</div>
<div align="center">
    <a href="https://xiaoqian-shen.github.io/" target="_blank">Xiaoqian Shen</a><sup>1</sup>,</span>
    <a href="https://wx-zhang.github.io/" target="_blank">Wenxuan Zhang</a><sup>1</sup>,</span>
    <a href="https://junchen14.github.io/" target="_blank">Jun Chen</a><sup>1,2</sup>,</span>
    <a href="https://cemse.kaust.edu.sa/profiles/mohamed-elhoseiny" target="_blank">Mohamed Elhoseiny</a><sup>1</sup></span>
</div>

<div align="center">
    <sup>1</sup>KAUST&emsp;
    <sup>2</sup>Meta AI&emsp;
</div>

## 🔍 Overview

![alt text](assets/teaser.jpg)

![alt text](assets/main.jpg)

## :rocket: Get Started

### ⚙️ Environment setup

```bash
git clone https://github.com/xiaoqian-shen/Vgent.git
cd Vgent
conda create -n vgent python==3.11
conda activate vgent
pip install -r requirements.txt
```
### 📊 Evaluation

Download evaluation benchmarks

+ [MLVU](https://huggingface.co/datasets/MLVU/MVLU)
+ [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME)
+ [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench)

Suggest to run on 8 A100 GPUs

Step1: Construct Offline Graph

```python 
torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port=$MASTER_PORT \
    -m vgent_graph \
    --model_name $model_name \ # qwenvl25_7b
    --task $task \ # mlvu, lvb, videomme
    --data_path $data_path \
```

Step2: Retrieval-Reasoning-Augmented Generation

```python
torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port=$MASTER_PORT \
    -m vgent_rag \
    --model_name $model_name \ # qwenvl25_7b
    --task $task \ # mlvu, lvb, videomme
    --data_path $data_path \
```

> [!Tip]
> You can add any LVLMs in Vgent pipeline by rewriting `load_video()`, `load_model()` and `mllm_response()`

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```BibTeX
@inproceedings{shen2025vgent,
  title={Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding},
  author={Shen, Xiaoqian and Zhang, Wenxuan and Chen, Jun and Elhoseiny, Mohamed},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
