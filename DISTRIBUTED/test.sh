
module load cesga/2022 python/3.10.8
source $STORE/hpc_tools/bert/bin/activate

BASE_PATH=$LUSTRE/hpc_tools/bert
mkdir -p $BASE_PATH/hf_home $BASE_PATH/models $BASE_PATH/checkpoints $BASE_PATH/logs

export HF_HOME=$BASE_PATH/hf_home
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

python3 train.py fit \
    --seed_everything=1234 \
    --trainer.devices=1 \
    --trainer.num_nodes=1 \
    --trainer.max_epochs=1 \
    --trainer.default_root_dir=$BASE_PATH/logs/test \
    --data.workers=16 \
    --data.train_subset=512 \
    --data.validation_subset=64 \
    --checkpoint.every_n_train_steps=100 \
    --checkpoint.dirpath=$BASE_PATH/checkpoints/test \
    --data.batch_size=16 \
    --trainer.fast_dev_run=5

