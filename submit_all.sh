#!/bin/env bash

sbatch job.sh baseline         --batch-size=16
sbatch job.sh big_batch        --batch-size=128
sbatch job.sh amp              --batch-size=16  --amp
sbatch job.sh compile_default  --batch-size=16  --compile=default
sbatch job.sh compile_overhead --batch-size=16  --compile=reduce-overhead
sbatch job.sh compile_autotune --batch-size=16  --compile=max-autotune
sbatch job.sh all              --batch-size=128 --compile=default --amp --tf32 --fused

