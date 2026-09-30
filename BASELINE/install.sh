#!/bin/env bash

module load cesga/2022 python/3.10.8
mkdir -p $STORE/hpc_tools
python -m venv $STORE/hpc_tools/bert
source $STORE/hpc_tools/bert/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .

exit 0

python -m pip install jupyter
python -m ipykernel install --user --name=bert
jupyter lab --ip `hostname -i`

exit 0

python -m pip install torch_tb_profiler
tensorboard --logdir=./log/baseline --host `hostname -i`
