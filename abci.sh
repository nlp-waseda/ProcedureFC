#!/bin/bash

#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gcc50435
#PBS -j oe

cd ${PBS_O_WORKDIR}

#module load cuda/12.8/12.8.1
source key.sh
source .venv/bin/activate
#CUDA_VISIBLE_DEVICES=0 python Main.py --vllm-default-bench
python Main.py --openai-default-bench