#!/bin/bash
#$ -cwd
#$-l node_q=1
#$-l h_rt=2:00:00
#$-j y

module load cuda/12.8.0
source key.sh
export HF_HOME="/gs/bs/tgh-25IAP/somatani/cache/huggingface"
export CUDA_VISIBLE_DEVICES=0
source .venv/bin/activate
python Main.py --vllm-default-bench 0 2