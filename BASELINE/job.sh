#!/bin/env bash

#SBATCH --job-name="BERT"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out

module load cesga/2022 python/3.10.8

source $STORE/hpc_tools/bert/bin/activate

mkdir -p $LUSTRE/hpc_tools/bert/hf_home
mkdir -p $LUSTRE/hpc_tools/bert/models
mkdir -p $LUSTRE/hpc_tools/bert/checkpoints
mkdir -p $LUSTRE/hpc_tools/bert/logs

export HF_HOME=$LUSTRE/hpc_tools/bert/hf_home
RUN_NAME=$1
shift

python3 train.py --workers=16 \
    --epochs=4 --train-subset=8192 --validation-subset=128 --save-steps=800 \
    --model=$LUSTRE/hpc_tools/bert/models/${RUN_NAME}.pt \
    --checkpoint=$LUSTRE/hpc_tools/bert/checkpoints/${RUN_NAME}.pt \
    --profiler=$LUSTRE/hpc_tools/bert/logs/${RUN_NAME} \
    "$@"

