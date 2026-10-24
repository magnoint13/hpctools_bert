#!/bin/env bash

#SBATCH --job-name="BERT"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

module load cesga/2022 python/3.10.8

source $STORE/hpc_tools/bert/bin/activate

BASE_PATH=$LUSTRE/hpc_tools/bert
mkdir -p $BASE_PATH/hf_home $BASE_PATH/models $BASE_PATH/checkpoints $BASE_PATH/logs

export HF_HOME=$BASE_PATH/hf_home
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

RUN_NAME=$1
shift

srun python3 train.py fit \
    --seed_everything=1234 \
    --trainer.devices=$SLURM_NTASKS_PER_NODE \
    --trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    --trainer.max_epochs=4 \
    --trainer.default_root_dir=$BASE_PATH/logs/${RUN_NAME} \
    --data.workers=16 \
    --data.train_subset=8192 \
    --data.validation_subset=128 \
    --checkpoint.every_n_train_steps=800 \
    --checkpoint.dirpath=$BASE_PATH/checkpoints/${RUN_NAME} \
    "$@"

