import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("logs")
GRAPH_OUT = Path("img")
GRAPH_OUT.mkdir(exist_ok=True)

SHOW_GRAPHS = True
ALL = True
ENABLED_GRAPHS = {
    "exec_time": False,
    "speedup": False,
    "scaling": False,
    "memory": False,
    "quality": False,
    "batch": True,
}

FILENAME_PATH = re.compile(r"(.+)_(\d)gpus_(\d+)")
VALIDATION_PAT = re.compile(r"\[epoch\s+(\d+)\]\s+validation\s+EM=([\d.]+)\s+F1=([\d.]+)")
MEMORY_PAT = re.compile(r"Memory used:\s*([\d.]+)\s*GB")
SAVE_TIME_PAT = re.compile(r"Final model saved in\s*([\d.]+)\s*s")
PROFILER_TOTAL_PAT = re.compile(r"\|\s*Total\s*\|\s*-\s*\|\s*\d+\s*\|\s*([\d.]*)\s*\|")
TRAINING_TIME_PAT = re.compile(
    r"\|\s*run_training_epoch\s*\|\s*"
    r"([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|"
)
BATCH_TIME_PAT = re.compile(
    r"\|\s*run_training_batch\s*\|\s*"
    r"[\d.]+\s*\|\s*(\d+)\s*\|"
)

@dataclass
class Log:
    benchmark: str = ""
    gpus: int = 0
    slurm_id: int = 0
    max_epochs: int = 0
    em: float = 0.0
    f1: float = 0.0
    memory: float = 0.0
    save_time: float = 0.0

    train_mean_time: float = 0.0
    train_total_time: float = 0.0
    num_batches: int = 0
    profiler_total_time: float = 0.0

    train_mean_speedup: float = 0.0
    train_total_speedup: float = 0.0
    profiler_speedup: float = 0.0

    def calc_speedup(self, baseline):
        self.train_total_speedup = baseline.train_total_time / self.train_total_time
        self.train_mean_speedup = baseline.train_mean_time / self.train_mean_time
        self.profiler_speedup = baseline.profiler_total_time / self.profiler_total_time

data = {}
for file in ROOT.glob("*.out"):
    log = Log()

    # ---- extract benchmark name, GPU count and SLURM_ID from filename ----
    file_match = FILENAME_PATH.fullmatch(file.stem)
    if not file_match:
        #print(f"Skipping unexpected filename: {file.name}", file=sys.stderr)
        continue
    log.benchmark, gpus, slurm_id = file_match.groups()
    log.gpus = int(gpus)
    log.slurm_id = int(slurm_id)

    # ---- read file ----
    file_content = file.read_text(errors="replace")

    # ---- get validation metrics ----
    validation = VALIDATION_PAT.findall(file_content)
    if not validation:
        print(f"No validation metrics found: {file.name}", file=sys.stderr)
        continue
    # (epoch_num, em, f1), then get last epoch
    validation = [
        (int(x[0]), float(x[1]), float(x[2]))
        for x in validation
    ]
    log.max_epochs, log.em, log.f1 = max(validation, key=lambda x: x[0])

    # ---- memory used ----
    log.memory = float(MEMORY_PAT.findall(file_content)[0])

    # ---- save time ----
    log.save_time = float(SAVE_TIME_PAT.findall(file_content)[0])

    # ---- training time ----
    train_time = TRAINING_TIME_PAT.findall(file_content)[0]
    log.train_mean_time, log.train_total_time = float(train_time[0]), float(train_time[1])

    # ---- total batches executed ----
    log.num_batches = int(BATCH_TIME_PAT.findall(file_content)[0])

    # ---- total time ----
    log.profiler_total_time = float(PROFILER_TOTAL_PAT.findall(file_content)[0])

    data[(log.benchmark, log.gpus)] = log

baseline = data[("baseline", 1)]
for x in data.values():
    x.calc_speedup(baseline)
    print(x)



# =================================================================================================
# ==== PLOTS ======================================================================================
# =================================================================================================


ORDER = [
    "baseline",
    "big_batch",
    "amp",
    "compile_default", "compile_overhead", "compile_autotune",
    "all_default", "all_overhead", "all_autotune",
]
benchmark_names = [b for b in ORDER if (b, 1) in data]
GPUS = (1, 2, 4)

def values(attr, gpus):
    return [getattr(data[(b, gpus)], attr) for b in benchmark_names]

# ---- quality vs time ----
if ALL or ENABLED_GRAPHS["batch"]:
    fig, ax = plt.subplots(figsize=(10, 7))
    cols = {f"{g} GPU{'s' if g > 1 else ''}": values("num_batches", g) for g in GPUS}
    res = ax.grouped_bar(cols, tick_labels=benchmark_names, orientation="horizontal")
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9)
    ax.margins(x=0.15)

    ax.set_xlabel("Batches")
    ax.set_title("Number of batches per GPU")
    ax.legend(loc="upper right")

    fig.savefig(GRAPH_OUT / "batch.svg", format="svg")

# batch_size es por GPU ==> entonces el batch global es 4 veces mas grande ==> menos pasos por epoch
if ALL or ENABLED_GRAPHS["exec_time"]:
    fig, ax = plt.subplots(figsize=(10, 7))
    cols = {f"{g} GPU{'s' if g > 1 else ''}": values("train_total_time", g) for g in GPUS}
    res = ax.grouped_bar(cols, tick_labels=benchmark_names, orientation="horizontal")
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9)
    ax.margins(x=0.15)

    ax.set_xlabel("Execution time (s)")
    ax.set_title("Total training time vs baseline (1 GPU)")
    ax.legend(loc="lower right")

    fig.savefig(GRAPH_OUT / "exec_time.svg", format="svg")

    # print this data as tables:
    print("| | 1 GPU | 2 GPUs | 4 GPUs |\n| - | - | - | - |")
    for i, bench in enumerate(benchmark_names):
        print(f"| `{bench}` | {cols['1 GPU'][i]:.2f}s | {cols['2 GPUs'][i]:.2f}s | {cols['4 GPUs'][i]:.2f}s |")


# ---- speedup ----
if ALL or ENABLED_GRAPHS["speedup"]:
    fig, ax = plt.subplots(figsize=(11, 7))
    cols = {f"{g} GPU{'s' if g > 1 else ''}": values("train_total_speedup", g) for g in GPUS}
    res = ax.grouped_bar(cols, tick_labels=benchmark_names, orientation="horizontal")
    for container in res.bar_containers:
        ax.bar_label(container, padding=3, fmt="{:.2f}x")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9)
    ax.margins(x=0.15)

    ax.set_xlabel("Speedup")
    ax.set_title("Total training speedup vs baseline (1 GPU)")
    ax.legend(loc="lower right")

    fig.savefig(GRAPH_OUT / "speedup.svg", format="svg")


# ---- scaling of the optimization ----
if ALL or ENABLED_GRAPHS["scaling"]:
    fig, ax = plt.subplots(figsize=(11, 7))

    group_names = []
    column_values = {"2 GPUs": [], "4 GPUs": []}
    for log in data.values():
        if log.gpus != 1:
            continue
        group_names.append(log.benchmark)
        two  = data[(log.benchmark, 2)]
        four = data[(log.benchmark, 4)]
        column_values["2 GPUs"].append(log.train_total_time / two.train_total_time)
        column_values["4 GPUs"].append(log.train_total_time / four.train_total_time)

    res = ax.grouped_bar(column_values, tick_labels=group_names, orientation="horizontal")
    for container in res.bar_containers:
        ax.bar_label(container, padding=3, fmt="{:.2f}x")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9)
    ax.margins(x=0.15)

    ax.set_xlabel("Speedup")
    ax.set_title("Scaling of the same optimization")
    ax.legend(loc="lower right")

    fig.savefig(GRAPH_OUT / "scaling.svg", format="svg")


# ---- memory usage ----
if ALL or ENABLED_GRAPHS["memory"]:
    fig, ax = plt.subplots(figsize=(11, 7))
    cols = {f"{g} GPU{'s' if g > 1 else ''}": values("memory", g) for g in GPUS}
    res = ax.grouped_bar(cols, tick_labels=benchmark_names, orientation="horizontal")
    for container in res.bar_containers:
        ax.bar_label(container, padding=3, fmt="{:.3f}")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, top=0.9)
    ax.margins(x=0.15)

    ax.set_xlabel("Memory usage (GB)")
    ax.set_title("Peak memory usage of a single GPU")
    ax.legend(loc="lower right")

    fig.savefig(GRAPH_OUT / "memory.svg", format="svg")

# ---- quality vs time ----
if ALL or ENABLED_GRAPHS["quality"]:
    fig, ax = plt.subplots(figsize=(11, 7))
    COLORS = ["green", "gray", "blue", "red", "black", "aqua", "purple", "orange", "pink"]
    for i, bench in enumerate(benchmark_names):
        one  = data[(bench, 1)]
        two  = data[(bench, 2)]
        four = data[(bench, 4)]

        x = [one.train_total_time, two.train_total_time, four.train_total_time]
        y = [one.f1, two.f1, four.f1]
        ax.plot(x, y, label=bench, color=COLORS[i])
        ax.scatter(x[0], y[0], marker="o", color=COLORS[i])
        ax.scatter(x[1], y[1], marker="+", color=COLORS[i])
        ax.scatter(x[2], y[2], marker="s", color=COLORS[i])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F1")
    ax.set_title("Quality vs Training time")

    legend1 = ax.legend(title="Benchmark", loc="lower right", bbox_to_anchor=(1, 0))
    gpu_legend = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="1 GPU"),
        Line2D([], [], color="black", marker="+", linestyle="None", label="2 GPUs"),
        Line2D([], [], color="black", marker="s", linestyle="None", label="4 GPUs"),
    ]
    legend2 = ax.legend(handles=gpu_legend, title="Configuration", loc="lower right", bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)

    fig.savefig(GRAPH_OUT / "quality.svg", format="svg")

if SHOW_GRAPHS:
    plt.show()

