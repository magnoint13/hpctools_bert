#!/bin/env bash

run() {
    local name=$2
    shift 2
    sbatch --job-name="${name}" job.sh "${name}" "$@"
}

# Important for SLURM output
mkdir -p logs

AMP="--trainer.precision=bf16-mixed"
ALL="--model.tf32=true --model.fused=true $AMP"

run baseline         --data.batch_size=16
run big_batch        --data.batch_size=128
run amp              --data.batch_size=16  $AMP
run compile_default  --data.batch_size=16  --model.compile=default
run compile_overhead --data.batch_size=16  --model.compile=reduce-overhead
run compile_autotune --data.batch_size=16  --model.compile=max-autotune
run all_default      --data.batch_size=128 --model.compile=default         $ALL
run all_overhead     --data.batch_size=128 --model.compile=reduce-overhead $ALL
run all_autotune     --data.batch_size=128 --model.compile=max-autotune    $ALL

