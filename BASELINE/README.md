# BERT-Base Fine-Tuning on SQuAD — Baseline Implementation

This project fine-tunes **BERT-base-uncased**
([google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased))
for extractive Question Answering on the **SQuAD** dataset
([rajpurkar/squad](https://rajpurkar.github.io/SQuAD-explorer/)),
using a single GPU.

The implementation uses the Hugging Face `transformers` and `datasets` libraries on top of PyTorch.

# How to use

This is a standard Python project,
you can use the following command to install the required dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

However, the most recent version of PyTorch uses a different CUDA version,
incompatible with the driver version available in the Finisterrae III
(where these benchmarks are run).
Therefore, a specific instalation script for this system is provided:

```sh
./install.sh
```

# Project structure

The code is mainly organized into four files.

`train.py` is the main training script in charge of
the data loading, training loop, checkpointing, timing/profiling and evaluation.
This script accepts some CLI arguments to customize the execution:

- `--batch-size`: size of the batch in samples
- `--max-length`: maximum number of tokens per example
  (question, context and special tokens) after tokenization,
  longer sequences are truncated.
  Capped at 512 due to BERT's positional embedding limit.
- `--epochs`: number of iterations over the training dataset.
- `--lr`: default learning rate given to the optimizer.
- `--workers`: number of processes used in the data loading.
- `--train-subset`: number of examples to use as the training dataset.
- `--validation-subset`: number of examples to use as the validation dataset.
- `--log-steps`: number of steps between logs.
- `--save-steps`: number of steps between checkpoints of the model.

File options:

- `--profiler`: directory where to store the profiler output
  (outputs 2 `.pt.trace.json` files between 13-22 MB each).
- `--checkpoint`: file where to store the checkpoints of the model (1.3 GB).
- `--model`: file where to store the weights of the model (416 MB).
- `--resume`: if the checkpoint file exists, starts the training from there.

Optimization options:

- `--compile`: uses `torch.compile()` on the model to achieve better performance,
  but the compilation takes time on the first step.
- `--amp`: activates mixed precision.
- `--tf32`: reduces the precision of matrix multiplications that remain in FP32.
- `--fused`: uses a fused AdamW optimizer.

The following steps are the operations that this script performs:

1.  **Model and tokenizer loading**:
    loads `BertForQuestionAnswering`
    (BERT-base with a QA head that outputs a *start* and *end* logit per token)
    and a `BertTokenizerFast`
    (needed because it provides character-to-token offset mappings,
    required to convert character-based answer positions into token positions).
    If enabled, also compiles the model and configures the optimizer (`AdamW`) accordingly.

1.  **Dataset loading and splitting**:
    loads SQuAD via `datasets.load_dataset`,
    then re-splits the original `train` split into a 90% train / 10% validation set
    using a fixed seed,
    and reserves the official SQuAD `validation` split as a held-out `test` set.

1.  **Tokenization**:
    using the helper module `token_model.py`,
    performs the tokenization and answer-span label generation of the examples.

1.  **Checkpointing and resume**:
    checks if the model state should be restored from a previous execution.

1.  **Training loop**:
    for every epoch and batch, executes a step of the training
    within a PyTorch profile context.

    Batches are transferred to the GPU asynchronously (non-blocking),
    and the loss is accumulated as a GPU tensor
    rather than being synchronized on every step.
    It is only converted to a Python float when it needs to be logged.

    `eval_model.py` implements the official SQuAD evaluation metrics:
    Exact Match (EM) and token-level F1,
    computed after normalizing text (lowercasing, removing articles and punctuation).
    It also implements a simplified best-span search over the model's start/end logits (`get_best_answer`),
    used during validation after each epoch.

    Checkpoints and evaluations are also measured and counted as overhead,
    which is later subtracted from the total execution time.

Additionally, the script `infer.py` is a simple interactive command-line inference prompt
on a trained model,
to manually test it.

## Complementary scripts

These scripts set up the required environment for the model to run in the Finisterrae III,
and therefore making these benchmarks repeatable.

- `install.sh`: initial script to set up the Python environment to execute the model
  in the supercomputer Finisterrae III.
- `test_installation.py`: simple python script to test that the required libraries are available.
- `job.sh`: SLURM job to launch the training in the Finisterrae III.
  It sets up the required environment for it to work properly.
- `submit_all.sh`: convenience script to create multiple jobs with different configurations,
  which allows to compare the effects of the applied optimizations.

**Hardware**:

| Parameter  | Value                                |
| -          | -                                    |
| CPU        | Intel Xeon Platinum 8352Y (32 cores) |
| GPU        | NVIDIA A100 40 GB                    |
| Filesystem | LUSTRE                               |

**Software versions**:

| Parameter    | Value       |
| -            | -           |
| Python       | 3.10.8      |
| PyTorch      | 2.8.0+cu128 |
| CUDA         | 12.8        |
| Transformers | 5.17.0      |
| Datasets     | 5.0.1       |

**Model configuration**:

| Parameter                | Value                           |
| -                        | -                               |
| Model                    | `google-bert/bert-base-uncased` |
| Dataset                  | SQuAD (`rajpurkar/squad`)       |
| Epochs                   | 4                               |
| Train examples used      | 8192                            |
| Validation examples used | 128                             |
| Batch size               | 16 or 128                       |
| Max sequence length      | 384                             |
| Learning rate            | 2e-5                            |

**Benchmark parameters**:

| Benchmark          | Batch size | Compile           | AMP | TF32 |
| -                  | -          | -                 | -   | -    |
| `baseline`         | 16         | No                | No  | No   |
| `big_batch`        | 128        | No                | No  | No   |
| `amp`              | 16         | No                | Yes | No   |
| `compile_default`  | 16         | `default`         | No  | No   |
| `compile_overhead` | 16         | `reduce-overhead` | No  | No   |
| `compile_autotune` | 16         | `max-autotune`    | No  | No   |
| `all_default`      | 128        | `default`         | Yes | Yes  |
| `all_overhead`     | 128        | `reduce-overhead` | Yes | Yes  |
| `all_autotune`     | 128        | `max-autotune`    | Yes | Yes  |


# Time and profiling report

## General observations

| Benchmark          | CPU time (MM:SS) | Job wall time (MM:SS) | Train time (s) | Overhead (s) | Memory used (GB) | EM (epoch 4) | F1 (epoch 4) |
| -                  | -                | -                     | -              | -            | -                | -            | -            |
| `baseline`         | 09:05            | 10:57                 | 504.82         | 14.34        |  5.34            | 0.5938       | 0.7310       |
| `big_batch`        | 08:21            | 10:14                 | 477.81         | 10.69        | 33.20            | 0.5234       | 0.6504       |
| `amp`              | 02:40            | 04:46                 | 117.26         | 30.45        |  4.10            | 0.6406       | 0.7649       |
| `compile_default`  | 09:50            | 11:21                 | 522.38         | 20.78        |  5.14            | 0.6562       | 0.7893       |
| `compile_overhead` | 10:02            | 11:30                 | 534.00         | 21.52        |  5.13            | 0.6016       | 0.7371       |
| `compile_autotune` | 16:54            | 13:06                 | 612.37         | 36.69        |  5.20            | 0.6172       | 0.7639       |
| `all_default`      | 02:42            | 04:19                 |  94.81         | 28.84        | 19.41            | 0.5078       | 0.6600       |
| `all_overhead`     | 02:38            | 04:19                 |  91.51         | 32.42        | 19.39            | 0.5312       | 0.6703       |
| `all_autotune`     | 06:54            | 06:00                 | 180.31         | 46.02        | 19.39            | 0.5625       | 0.7020       |

Below are the performance and model-quality metrics
for each of the nine benchmark configurations described earlier.

- **CPU time** and **job wall time** are reported by SLURM's `seff`.
  CPU time represents the accumulated CPU time used by the job,
  whereas wall time represents the real elapsed time from job start to job completion.
- **Train time** is the time measured for the training loop itself.
  The original end-to-end measurement also includes validation and checkpointing,
  which are reported separately as **overhead**.
- **Memory used** is the peak GPU memory allocated during the run,
  measured with `torch.cuda.max_memory_allocated()`.
  This is particularly relevant when evaluating the effect of increasing the batch size.

Wall time is consistently higher than train `time + overhead`,
by roughly 125-140s across almost every run.
That gap is everything outside the measured loop:
environment setup, downloading the model and tokenizer, and tokenizing the dataset.
Since it doesn't scale with batch size or depend on the compile mode,
it behaves as a fixed cost every job pays regardless of configuration.

CPU time is usually lower than wall time,
since most of the work happens on the GPU while the CPU sits idle.
compile_autotune is the exception:
its CPU time (16:54) is higher than its wall time (13:06).
This is because `max-autotune` benchmarks use Triton kernels in parallel
across multiple CPU threads,
so SLURM's aggregated core-time counter can exceed the wall-clock duration.

**Batch size**:

With `batch_size=16`, the training runs for 2048 steps,
whereas with `batch_size=128` it requires only 256 steps.
Therefore, 
`big_batch` performs 8x fewer optimizer updates than the baseline.
Training time drops modestly, but GPU memory jumps from 5.34GB to 33.20GB.

That difference in update count also explains the EM/F1 numbers:
`big_batch` scores clearly lower than baseline (0.52 vs 0.59 EM),
but that isn't evidence the larger batch hurts quality
(it's just 8x fewer gradient updates for the same number of epochs).

Any batch-size comparison in this table should be read as a speed comparison,
not a quality comparison,
unless the learning rate or the step count is adjusted to compensate.

**Mixed precision**:

The most significant reduction in training time comes from mixed-precision training.
The `amp` configuration reduces the training time from 504.82 s to 117.26 s,
corresponding to approximately a **4.31x speedup**.

The profiler provides evidence for the change in GPU execution.
In the baseline configuration,
the main CUDA hotspots are matrix-multiplication operations:
`aten::mm` accounts for 54.38% of the self CUDA time
and `aten::addmm` for another 27.16%.

With AMP enabled, those same operations run through BF16 tensor-core kernels instead
(`ampere_bf16_s16816gemm_...`), along with BF16 attention kernels.
That lines up with the A100's spec sheet:
about 19.5 TFLOPS on plain FP32 vs. 312 TFLOPS on BF16 tensor cores.
This is over 15x more theoretical throughput on the same chip.

AMP also lowers peak memory here, from 5.34GB to 4.10GB,
since activations and gradients take up half the space.

**torch.compile**:

The `torch.compile` configurations do not reduce the training time in isolation in this experiment.
`compile_default` (522.38s), `compile_overhead` (534.00s) and `compile_autotune` (612.37s)
are all slower than the uncompiled baseline.

The profiler shows that the compiled configurations execute through compiled regions
such as `CompiledFunction` and `CompiledFunctionBackward`,
together with generated Triton kernels.
In the `compile_default` configuration,
for example, `aten::mm` accounts for 66.67% of the self CUDA time,
while `CompiledFunctionBackward` accounts for approximately 454 ms of CUDA time
over the three profiled steps.
The `compile_overhead` and `compile_autotune` profiles
similarly show substantial execution inside compiled backward regions.

So compilation is clearly changing how the model executes;
it just doesn't translate into a net win here.
`max-autotune` comes out slowest of the three,
consistent with the extra time it spends benchmarking kernel candidates before picking one.

**Combining everything**:

Finally, combining the larger batch size, mixed precision, and compilation
produces the lowest training times in the benchmark:

- `all_default`: **94.81 s**, corresponding to approximately **5.32x** speedup over the baseline.
- `all_overhead`: **91.51 s**, corresponding to approximately **5.52x** speedup over the baseline.
- `all_autotune`: **180.31 s**, corresponding to approximately **2.80x** speedup over the baseline.

Memory sits around 19.4GB for all three,
well below the 33.2GB of big_batch alone,
since AMP's smaller activations partly offset the extra memory the larger batch needs.

The profiler confirms the combined runs behave differently from any single optimization on its own.
The use of BF16 GEMM and attention kernels,
Triton-fused regions,
and compiled forward/backward blocks all show up together
(`CompiledFunctionBackward` runs at ~525ms of CUDA time in `all_autotune`, for instance).
Therefore, the speedup comes from all of the techniques acting together.

## Profiler analysis

The profiler was used to analyse the GPU execution during a small number of training steps
rather than the complete training job,
so aren't directly comparable to the end-to-end times above.

The profiler is most useful here for identifying where the computation is spent
rather than for measuring the complete training duration.
The end-to-end timings in the first table
should be used to evaluate the actual performance of each benchmark,
while the profiler explains the changes in the underlying GPU execution.

The profiler output is based on `torch.profiler.key_averages()`,
which aggregates events by operation.

The main observations are:

| Benchmark          | Main profiler observations                                                                | Interpretation                                                                                                                                    |
| -                  | -                                                                                         | -                                                                                                                                                 |
| `baseline`         | `aten::mm`: 54.38%; `aten::addmm`: 27.16%; attention backward: 6.01%                      | Matrix multiplications dominate the profiled CUDA workload.                                                                                       |
| `big_batch`        | `aten::mm`: 56.25%; `aten::addmm`: 27.84%; attention backward: 6.27%                      | The same GEMM-heavy structure remains, but each profiled step performs substantially more work.                                                   |
| `amp`              | BF16 GEMM and BF16 attention kernels appear                                               | Mixed precision changes the GPU kernels used for the main compute operations.                                                                     |
| `compile_default`  | `aten::mm`: 66.67%; `CompiledFunctionBackward`: ~454 ms CUDA                              | A significant part of the execution is represented by compiled regions.                                                                           |
| `compile_overhead` | `CompiledFunctionBackward`: ~463 ms CUDA                                                  | Similar compiled execution, with no end-to-end training-time improvement over baseline.                                                           |
| `compile_autotune` | `CompiledFunctionBackward`: ~443 ms CUDA                                                  | Autotuning changes the generated/selected execution, but its overall cost makes this configuration slower end-to-end.                             |
| `all_default`      | Compiled regions together with BF16/Triton kernels                                        | The combined configuration substantially changes the execution pattern.                                                                           |
| `all_overhead`     | Compiled/Triton and BF16 execution                                                        | This configuration obtains the lowest training-loop time in the benchmark.                                                                        |
| `all_autotune`     | `CompiledFunctionBackward`: \~525 ms; compiled region: \~255 ms; BF16 attention: \~239 ms | The profiler shows extensive use of compiled, Triton, and BF16 kernels, but the end-to-end time is higher than the other combined configurations. |

Overall, the profiling results are consistent with the timing measurements:
the baseline is dominated by matrix multiplications,
mixed precision replaces the main FP32 GEMM/attention kernels with BF16-capable kernels,
and `torch.compile` introduces compiled and Triton-generated regions.

