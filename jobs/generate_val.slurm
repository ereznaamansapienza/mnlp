#!/bin/bash
#SBATCH --job-name=mnlp-generate-val
#SBATCH --account=IscrC_MNLP26
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load python

cd /leonardo_work/IscrC_MNLP26/apromots_doubleN/hw2/ft_code

mkdir -p logs
mkdir -p /leonardo_work/IscrC_MNLP26/apromots_doubleN/hf_cache

export HF_HOME=/leonardo_work/IscrC_MNLP26/apromots_doubleN/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRITON_CACHE_DIR=$HF_HOME/triton

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source /leonardo/home/userexternal/apromots/.env/bin/activate

python -u scripts/generate.py \
  --config configs/train_qwen3_06b_lora.yaml \
  --split val \
  --setting rag
