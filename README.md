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
- `--max-length` <!-- TODO: explicar -->
- `--epochs`: number of iterations over the training dataset.
- `--lr`: default learning rate given to the optimizer.
- `--workers`: number of processes used in the data loading.
- `--train-subset`: number of examples to use as a the training dataset.
- `--validation-subset`: number of examples to use as a the validation dataset.
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
- `--tf32`: reduces the precision of some floating point operations, allowing it to run faster.
  Only makes sense when `--amp` is used.
- `--fused`: uses a fused AdamW optimizer.

The following steps are the operations that this script performs:

1.  **Model and tokenizer loading**:
    loads `BertForQuestionAnswering`
    (BERT-base with a QA head that outputs a *start* and *end* logit per token)
    and a `BertTokenizerFast`
    (needed because it provides character-to-token offset mappings,
    required to convert character-based answer positions into token positions).
    If enabled, also compiles de model and configures the optimizer (`AdamW`) accordingly.

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
    in a pytorch profile environment.

    The batches are sent to the GPU in a asynchronous non-blocking manner,
    as the results coming from the model.

    During this training loop,
    every epoch the model is evaluated using the Exact Match and F1 metrics.
    These are implemented in the `eval_model.py` module.

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
- `test_instalation.py`: simple python script to test that the required libraries are available.
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

