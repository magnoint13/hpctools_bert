#!/bin/env bash

module load cesga/2022 python/3.10.8
mkdir -p $STORE/hpc_tools
python -m venv $STORE/hpc_tools/bert
source $STORE/hpc_tools/bert/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .

