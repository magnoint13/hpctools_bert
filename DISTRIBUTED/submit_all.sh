#!/bin/env bash

run() {
    local gpus=$1
    local name=$2
    shift 2
    sbatch --job-name="${name}_${gpus}gpus" --ntasks-per-node=$gpus --gres=gpu:a100:$gpus \
        job.sh "${name}_${gpus}gpus" "$@"
}

# Important for SLURM output
mkdir -p logs

AMP="--trainer.precision=bf16-mixed"
ALL="--model.tf32=true --model.fused=true $AMP"

for g in 1 2; do
    run $g baseline         --data.batch_size=16
    run $g big_batch        --data.batch_size=128
    run $g amp              --data.batch_size=16  $AMP
    run $g compile_default  --data.batch_size=16  --model.compile=default
    run $g compile_overhead --data.batch_size=16  --model.compile=reduce-overhead
    run $g compile_autotune --data.batch_size=16  --model.compile=max-autotune
    run $g all_default      --data.batch_size=128 --model.compile=default         $ALL
    run $g all_overhead     --data.batch_size=128 --model.compile=reduce-overhead $ALL
    run $g all_autotune     --data.batch_size=128 --model.compile=max-autotune    $ALL
done
