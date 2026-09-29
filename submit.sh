#!/bin/env bash

#SBATCH --job-name="opt_baseline"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=log/%x_%j.out

module load cesga/2022 python/3.10.8

source $STORE/hpc_tools/bert/bin/activate

mkdir -p $LUSTRE/hpc_tools/bert/hf_home
mkdir -p $LUSTRE/hpc_tools/bert/models
mkdir -p $LUSTRE/hpc_tools/bert/checkpoints
mkdir -p $LUSTRE/hpc_tools/bert/logs

NAME=${SLURM_JOB_NAME:-baseline}
export HF_HOME=$LUSTRE/hpc_tools/bert/hf_home

# 32 workers might compete with the main task,
# lower it to 16 to avoid the start overhead.
python3 train.py --resume --compile="default" \
    --batch-size=128 --epochs=2 --workers=16 \
    --train-subset=10000 --validation-subset=200 \
    --model=$LUSTRE/hpc_tools/bert/models/$NAME.pt \
    --checkpoint=$LUSTRE/hpc_tools/bert/checkpoints/$NAME.pt \
    --profiler=$LUSTRE/hpc_tools/bert/logs/$NAME

