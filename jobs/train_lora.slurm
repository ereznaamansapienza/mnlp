#!/bin/bash
#SBATCH --job-name=mnlp-qwen3-rag-lora
#SBATCH --account=IscrC_MNLP26
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs
mkdir -p /leonardo_work/IscrC_MNLP26/apromots_doubleN/hf_cache

export HF_HOME=/leonardo_work/IscrC_MNLP26/apromots_doubleN/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TRITON_CACHE_DIR=$HF_HOME/triton

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source /leonardo/home/userexternal/apromots/.env/bin/activate

python -u scripts/train_lora.py --config configs/train_qwen3_06b_lora.yaml
