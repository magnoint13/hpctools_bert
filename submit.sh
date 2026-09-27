#!/bin/env bash

#SBATCH --job-name="BERT"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out

module load cesga/2022 python/3.10.8
source $STORE/hpc_tools/bert/bin/activate
python3 train.py --resume --batch-size=128 --epochs=1 --workers=32 --train-subset=2000 --validation-subset=200

