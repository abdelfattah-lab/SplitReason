import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def model_size(name: str) -> float | None:
    """Extract the numeric 'X' in '* X B *' → float(X)."""
    m = re.search(r"(\d+(\.\d+)?)B", name)
    return float(m.group(1)) if m else None


FOLDER_MAP: Dict[str, str] = {
    "32b_only": "32B",
    "14b_only": "14B",
    "APR_17_RANDOMSWITCH_10PC": "SpecR Random 10%",
    "APR20_SPECR_FULLTRAINGRPO_14B": "SpecR 14B",
    "15b_only": "1.5B",
    "APR_17_RANDOMSWITCH_5PC": "SpecR Random 5%",
    "APR20_SPECR_FULLTRAINGRPO_8B": "SpecR 8B",
    "8b_only": "8B",
    "APR20_SPECR_FULLTRAINGRPO": "SpecR 32B",
    "APR20_SPECR_MINIMALOFFLOAD": "SpecR 32B Minimal",
}

OPEN_TAG, CLOSE_TAG = "<bigmodel>", "</bigmodel>"

def char_mask(text: str,
              open_tag: str = OPEN_TAG,
              close_tag: str = CLOSE_TAG) -> np.ndarray:
    mask: List[int] = []
    inside = False
    i = 0
    L_open, L_close = len(open_tag), len(close_tag)

    while i < len(text):
        if text.startswith(open_tag, i):
            inside = True
            i += L_open
            continue
        if text.startswith(close_tag, i):
            inside = False
            i += L_close
            continue
        mask.append(1 if inside else 0)
        i += 1

    return np.asarray(mask, dtype=np.uint8)


def process_run(run_dir: Path) -> Dict[str, Any]:
    run_data: Dict[str, Any] = {}

    for idx, res_path in enumerate(run_dir.rglob("results_*.json")):
        stem = res_path.stem
        ts = res_path.stem.split("results_", 1)[1]
        matches = list(res_path.parent.glob(f"samples_*_{ts}.jsonl"))
        if not matches:
            print(f"⚠️  No samples file for {res_path.name}")
            continue

        samp_path = matches[0]        
        with open(res_path, "r") as f:
            res_json = json.load(f)
        exact_matches = [
            task_block["exact_match,none"]
            for task_block in res_json["results"].values()
            if "exact_match,none" in task_block
        ]
        exact_match_value = float(np.mean(exact_matches)) if exact_matches else None

        eval_time = float(res_json.get("total_evaluation_time_seconds", "nan"))
        offload_percent_list: List[float] = []
        if samp_path.exists():
            with open(samp_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    text = (
                        item.get("resps")[0][0]
                    )
                    mask = char_mask(text)
                    if mask.size:
                        offload_percent_list.append(float(mask.mean() * 100.0))
        run_data[idx] = {
            "accuracy": exact_match_value,
            "eval_time": eval_time,
            "offload_percent": offload_percent_list,
        }
    return run_data


def collect_all(base_path: str) -> Dict[str, Any]:
    base = Path(base_path).expanduser()
    aggregated: Dict[str, Any] = {}
    for folder, readable in FOLDER_MAP.items():
        run_dir = base / folder
        if not run_dir.is_dir():
            print(f"⚠️  Skipping missing directory: {run_dir}")
            continue
        aggregated[readable] = process_run(run_dir)
    return aggregated


if False:
    ROOT = "./"
    all_metrics = collect_all(ROOT)

    import pprint, json, datetime
    pprint.pprint(all_metrics, compact=True)

    out_file = f"aggregated_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nSaved → {out_file}")

latest_file = max(
    glob.glob("aggregated_metrics_*.json"),
    key=os.path.getctime
)
with open(latest_file, "r") as fh:
    all_metrics = json.load(fh)

def graph_type_1():
    systems            = []
    scatter_tuples     = []
    ignore_systems = [
        "SpecR Random 10%",
        "SpecR Random 5%",
        "SpecR 32B Minimal",
    ]

    for system_name, runs in all_metrics.items():
        if system_name in ignore_systems:
            continue

        eval_times = [r["eval_time"] for r in runs.values()]
        accuracies = [r["accuracy"] for r in runs.values()]

        # accumulate per‑run points for scatter
        for et, acc in zip(eval_times, accuracies):
            scatter_tuples.append(
                (system_name, et, acc * 100.0, "SpecR" in system_name)
            )

        systems.append(
            {
                "name":        system_name,
                "mean_time":   float(np.mean(eval_times)),
                "median_time": float(np.median(eval_times)),   # ← NEW ❶
                "accuracies":  [a * 100.0 for a in accuracies],   # convert to %
            }
        )

    median_lookup = {d["name"]: d["median_time"] for d in systems}
    systems.sort(key=lambda d: d["mean_time"])

    labels          = [d["name"]       for d in systems]
    accuracy_groups = [d["accuracies"] for d in systems]
    median_latencies  = [d["median_time"]  for d in systems]   # ← NEW ❷
    x_pos           = np.arange(len(labels))

    fig, (ax_box, ax_scat) = plt.subplots(
        1, 2, figsize=(21, 7), sharey=False
    )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.96,
                        bottom=0.12, wspace=0.05)

    ax_box.set_xlim(-0.6, len(labels) - 0.4)
    ax_box.tick_params(axis="x", labelsize=18)

    ax_lat = ax_box.twinx()
    ax_lat.plot(
        x_pos,
        median_latencies,
        "k--",
        linewidth=1.5,
        marker="o",
        label="Median eval. time",
    )
    ax_lat.set_ylabel("Evaluation time (seconds)", fontsize=22)
    ax_lat.tick_params(axis="y", labelsize=18)
    ax_lat.grid(False)
    ax_scat.margins(x=0.08)
    ax_box.boxplot(
        accuracy_groups,
        positions     = x_pos,
    )

    ax_box.set_ylabel("Accuracy (%)", fontsize=22)
    ax_box.set_xticks(x_pos)
    ax_box.set_xticklabels(labels, rotation=30, ha="center", fontsize=18)
    ax_box.tick_params(axis="y", labelsize=18)
    ax_box.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map   = {}

    baseline_shades = {1.5: "#ffb3b3",
                    8  : "#ff6666",
                    14 : "#ff1a1a",
                    32 : "#990000"}
    spec_shades     = {1.5: "#b3d9ff",
                    8  : "#66a3ff",
                    14 : "#3385ff",
                    32 : "#005ce6"}
    scatter_handles = {}

    # Simulated
    small_gain_factors = {8: 0.894,  14: 0.795, 32: 0.729}

    def model_size(name: str):
        import re
        m = re.search(r"(\d+(\.\d+)?)B", name)
        return float(m.group(1)) if m else None
    for name, et, acc, is_spec in scatter_tuples:
        sz = model_size(name)
        if sz is None or sz not in baseline_shades:
            continue

        colour = spec_shades[sz] if is_spec else baseline_shades[sz]
        marker = "^" if is_spec else "o"
        msize  = 250 if is_spec else 150
        if is_spec:
            sz = model_size(name)
            if sz in small_gain_factors:
                et = median_lookup["1.5B"] / small_gain_factors[sz]

        h = ax_scat.scatter(
            et, acc,
            marker       = marker,
            s            = msize,
            color        = colour,
            alpha        = 0.85,
        )
        scatter_handles[name] = h

    def model_size(name: str):
        import re
        m = re.search(r"(\d+(\.\d+)?)B", name)
        return float(m.group(1)) if m else None

    legend_rows = []

    for sys_name, artist in scatter_handles.items():
        sz = model_size(sys_name)
        if sz is None:
            continue
        legend_rows.append((
            "SpecR" in sys_name,
            -sz,
            artist,
            sys_name,
        ))

    legend_rows.sort()
    handles = [r[2] for r in legend_rows]
    labels  = [r[3].replace("SpecR", "SplitReasoner") for r in legend_rows]

    ax_scat.legend(handles, labels,
                fontsize=18, loc="lower right", framealpha=0.9, ncol=2)
    ax_scat.set_xlabel("Pipelined Simulation Evaluation Time (seconds)", fontsize=22)
    ax_scat.tick_params(axis="x", labelsize=18)
    ax_scat.tick_params(axis="y", labelsize=18)
    ax_scat.set_ylabel("Accuracy (%)", fontsize=22)

    fig.tight_layout()
    fig.savefig("acc_lat_separate.pdf", format="pdf")


def graph_type_2():
    systems            = []
    scatter_tuples     = []
    ignore_systems = [
        "SpecR Random 10%",
        "SpecR Random 5%",
        "SpecR 32B Minimal",
    ]

    for system_name, runs in all_metrics.items():
        if system_name in ignore_systems:
            continue

        eval_times = [r["eval_time"] for r in runs.values()]
        accuracies = [r["accuracy"] for r in runs.values()]

        # accumulate per‑run points for scatter
        for et, acc in zip(eval_times, accuracies):
            scatter_tuples.append(
                (system_name, et, acc * 100.0, "SpecR" in system_name)
            )

        systems.append(
            {
                "name":        system_name,
                "mean_time":   float(np.mean(eval_times)),
                "median_time": float(np.median(eval_times)),   # ← NEW ❶
                "accuracies":  [a * 100.0 for a in accuracies],   # convert to %
            }
        )

    median_lookup = {d["name"]: d["median_time"] for d in systems}
    systems.sort(key=lambda d: d["mean_time"])

    labels          = [d["name"]       for d in systems]
    accuracy_groups = [d["accuracies"] for d in systems]
    median_latencies  = [d["median_time"]  for d in systems]   # ← NEW ❷
    x_pos           = np.arange(len(labels))

    fig, (ax_box, ax_scat) = plt.subplots(
        1, 2, figsize=(21, 7), sharey=False
    )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.96,
                        bottom=0.12, wspace=0.05)

    ax_box.set_xlim(-0.6, len(labels) - 0.4)
    ax_box.tick_params(axis="x", labelsize=18)

    ax_lat = ax_box.twinx()
    ax_lat.plot(
        x_pos,
        median_latencies,
        "k--",
        linewidth=1.5,
        marker="o",
        label="Median eval. time",
    )
    ax_lat.set_ylabel("Evaluation time (seconds)", fontsize=22)
    ax_lat.tick_params(axis="y", labelsize=18)
    ax_lat.grid(False)
    ax_scat.margins(x=0.08)
    ax_box.boxplot(
        accuracy_groups,
        positions     = x_pos,
    )

    ax_box.set_ylabel("Accuracy (%)", fontsize=22)
    ax_box.set_xticks(x_pos)
    ax_box.set_xticklabels(labels, rotation=30, ha="center", fontsize=18)
    ax_box.tick_params(axis="y", labelsize=18)
    ax_box.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map   = {}

    baseline_shades = {1.5: "#ffb3b3",
                    8  : "#ff6666",
                    14 : "#ff1a1a",
                    32 : "#990000"}
    spec_shades     = {1.5: "#b3d9ff",
                    8  : "#66a3ff",
                    14 : "#3385ff",
                    32 : "#005ce6"}
    scatter_handles = {}

    # Simulated
    small_gain_factors = {8: 0.894,  14: 0.795, 32: 0.729}

    # ‑‑‑‑‑‑ Gather per‑system summary statistics ‑‑‑‑‑‑
    summary_rows   = []          # for plotting
    scatter_handles = {}         # for legend building

    for system_name, runs in all_metrics.items():
        if system_name in ignore_systems:
            continue

        eval_times = np.asarray([r["eval_time"]  for r in runs.values()],  dtype=float)
        accuracies = np.asarray([r["accuracy"]   for r in runs.values()], dtype=float) * 100.0

        mean_time_nonpipelined = eval_times.mean()
        std_time  = eval_times.std(ddof=1)                     # latency error bar (x‑err)
        mean_acc  = accuracies.mean()
        std_acc   = accuracies.std(ddof=1)                     # accuracy error bar (y‑err)

        is_spec   = "SpecR" in system_name
        sz        = model_size(system_name)

        # --- optional latency “speed‑up” simulation for SpecR models --------------
        if is_spec and sz in small_gain_factors:
            mean_time = median_lookup["1.5B"] / small_gain_factors[sz]
        else:
            mean_time = mean_time_nonpipelined

        summary_rows.append(
            {
                "name":        system_name,
                "nonpipelined_time": mean_time_nonpipelined,
                "mean_time":   mean_time,
                "std_time":    std_time,
                "mean_acc":    mean_acc,
                "std_acc":     std_acc,
                "is_spec":     is_spec,
                "size":        sz,
            }
        )

    # ‑‑‑‑‑‑ Plot one *mean* point per model with ±1 σ error bars ‑‑‑‑‑‑
    for row in summary_rows:
        colour = (spec_shades if row["is_spec"] else baseline_shades).get(row["size"])
        marker = "^" if row["is_spec"] else "o"
        msize  = 12 if row["is_spec"] else 10

        h = ax_scat.errorbar(
            row["mean_time"], row["mean_acc"],
            xerr=row["std_time"], yerr=row["std_acc"],
            fmt=marker, markersize=msize, markeredgewidth=0,
            ecolor=colour, color=colour, capsize=5, alpha=0.9,
            label=row["name"]        # each handle is unique for legend
        )
        scatter_handles[row["name"]] = h
    # ‑‑‑‑‑‑ Build a tidy legend (same sorting trick as before) ‑‑‑‑‑‑
    legend_rows = [
        ("SpecR" in n, -model_size(n), h[0], n.replace("SpecR", "SplitReasoner"))
        for n, h in scatter_handles.items()
    ]
    legend_rows.sort()

    handles = [t[2] for t in legend_rows]
    labels  = [t[3] for t in legend_rows]

    ax_scat.legend(
        handles, labels, fontsize=18,
        loc="lower right", framealpha=0.9, ncol=2
    )

    ax_scat.set_xlabel("Pipelined Simulation Time (seconds)", fontsize=22)
    ax_scat.set_ylabel("Accuracy (%)", fontsize=22)
    ax_scat.tick_params(axis="both", labelsize=18)
    ax_scat.margins(x=0.08)

    fig.tight_layout()
    fig.savefig("acc_lat_separate2.pdf", format="pdf")


# graph_type_2()
def graph_type_3():
    systems        = []
    scatter_tuples = []
    ignore_systems = [
        "SpecR Random 10%",
        "SpecR Random 5%",
        "SpecR 32B Minimal",
    ]

    # ─── 0️⃣  Gather per‑run raw numbers ─────────────────────────────────────
    for system_name, runs in all_metrics.items():
        if system_name in ignore_systems:
            continue

        eval_times = [r["eval_time"] for r in runs.values()]
        accuracies = [r["accuracy"]  for r in runs.values()]

        for et, acc in zip(eval_times, accuracies):
            scatter_tuples.append(
                (system_name, et, acc * 100.0, "SpecR" in system_name)
            )

        systems.append(
            {
                "name":        system_name,
                "mean_time":   float(np.mean(eval_times)),
                "median_time": float(np.median(eval_times)),
                "accuracies":  [a * 100.0 for a in accuracies],
            }
        )

    median_lookup   = {d["name"]: d["median_time"] for d in systems}
    systems.sort(key=lambda d: d["mean_time"])

    baseline_shades = {1.5: "#ffb3b3",  8: "#ff6666", 14: "#ff1a1a", 32: "#990000"}
    spec_shades     = {1.5: "#b3d9ff",  8: "#66a3ff", 14: "#3385ff", 32: "#005ce6"}

    # Simulated “speed‑up” factors for SpecR sizes
    small_gain_factors = {8: 0.894, 14: 0.795, 32: 0.729}

    # ─── 1️⃣  Summary stats per system ───────────────────────────────────────
    summary_rows = []
    for system_name, runs in all_metrics.items():
        if system_name in ignore_systems:
            continue

        eval_times = np.asarray([r["eval_time"] for r in runs.values()],  dtype=float)
        accuracies = np.asarray([r["accuracy"]  for r in runs.values()], dtype=float) * 100.0

        mean_time_nonpipelined = eval_times.mean()
        std_time = eval_times.std(ddof=1)
        mean_acc = accuracies.mean()
        std_acc  = accuracies.std(ddof=1)

        is_spec = "SpecR" in system_name
        sz      = model_size(system_name)

        if is_spec and sz in small_gain_factors:
            mean_time = median_lookup["1.5B"] / small_gain_factors[sz]
        else:
            mean_time = mean_time_nonpipelined

        summary_rows.append(
            {
                "name":  system_name,
                "mean_acc":            mean_acc,
                "std_acc":             std_acc,
                "mean_time":           mean_time,
                "std_time":            std_time,
                "nonpipelined_time":   mean_time_nonpipelined,
                "is_spec":             is_spec,
                "size":                sz,
            }
        )

    # ─── 2️⃣  ONE‑panel figure (scatter only) ────────────────────────────────
    fig, ax_scat = plt.subplots(figsize=(10, 7))

    scatter_handles = {}
    for row in summary_rows:
        colour = (spec_shades if row["is_spec"] else baseline_shades).get(row["size"])
        marker = "^" if row["is_spec"] else "o"
        msize  = 12 if row["is_spec"] else 10

        h = ax_scat.errorbar(
            row["mean_time"], row["mean_acc"],
            xerr=row["std_time"], yerr=row["std_acc"],
            fmt=marker, markersize=2*msize, markeredgewidth=0,
            ecolor=colour, color=colour, capsize=5, alpha=0.9,
            label=row["name"],
        )
        scatter_handles[row["name"]] = h

    # tidy legend
    legend_rows = [
        ("SpecR" in n, -model_size(n), h[0], n.replace("SpecR", "SplitReasoner"))
        for n, h in scatter_handles.items()
    ]
    legend_rows.sort()
    ax_scat.legend(
        [t[2] for t in legend_rows],
        [t[3] for t in legend_rows],
        fontsize=20, loc="lower right", framealpha=0.9, ncol=2
    )

    ax_scat.set_xlabel("Pipelined Simulation Time (seconds)", fontsize=22)
    ax_scat.set_ylabel("AIME24 Accuracy (%)", fontsize=22)
    # ax_scat.set_xscale("log")
    # set x axis lower limit to 0
    ax_scat.set_xlim(left=0)
    ax_scat.tick_params(axis="both", labelsize=18)
    ax_scat.margins(x=0.08)

    fig.tight_layout()
    fig.savefig("acc_lat_scatter.pdf", format="pdf")

    # ─── 3️⃣  Write LaTeX table (no on‑figure rendering) ─────────────────────
    latex_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Accuracy (\%) & Non-Pipelined (s) & Pipelined (s)\\",
        r"\midrule",
    ]
    for row in summary_rows:
        model = row["name"]
        acc   = f"{row['mean_acc']:.1f}"
        nonp  = f"{row['nonpipelined_time']:.0f}" if row["is_spec"] else "–"
        pipe  = f"{row['mean_time']:.0f}"
        latex_lines.append(f"{model} & {acc} & {nonp} & {pipe}\\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}"]

    with open("model_metrics_table.tex", "w") as fh:
        fh.write("\n".join(latex_lines))
    print("Saved: acc_lat_scatter.pdf  |  model_metrics_table.tex")

def graph_teaser() -> None:
    """
    One point per *model size*:
        • circles  – plain models   (1.5 B, 8 B, 14 B)
        • stars    – SpecR models   (8 B, 14 B, 32 B)

    Two grey dashed lines connect the sizes within each group.
    """
    ignore_systems = {
        "SpecR Random 10%", "SpecR Random 5%", "SpecR 32B Minimal", "32B"
    }

    # --- mean values --------------------------------------------------------
    plain_models  = {}
    spec_models   = {}

    for sys_name, runs in all_metrics.items():
        if sys_name in ignore_systems:
            continue

        mean_time = np.mean([r["eval_time"] for r in runs.values()])
        mean_acc  = np.mean([r["accuracy"]  for r in runs.values()]) * 100.0  # %

        if sys_name.startswith("SpecR"):
            spec_models[sys_name]  = (mean_time, mean_acc)
        else:
            plain_models[sys_name] = (mean_time, mean_acc)

    # sort by numeric model size (1.5, 8, 14, 32)
    def sort_key(name: str) -> float:
        m = re.search(r"(\d+(\.\d+)?)B", name)
        return float(m.group(1)) if m else 0.0

    plain_items = sorted(plain_models.items(), key=lambda p: sort_key(p[0]))
    spec_items  = sorted(spec_models.items(),  key=lambda p: sort_key(p[0]))

    # ─── colour palettes ──────────────────────────────────────────
    baseline_shades = {          # 1.5 B, 8 B, 14 B
        14: "#213448",   # darkest blue‑grey
        8:   "#547792",   # medium slate‑blue
        1.5:  "#94B4C1",   # light steel‑blue
    }

    spec_shades = {              # 8 B, 14 B, 32 B
        8:  "#88304E",    # vivid red‑pink
        14: "#522546",    # deep raspberry
        32: "#F7374F",    # dark wine
    }
    def size_of(name: str) -> float:
        return sort_key(name)

    # --- draw ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    # circles  – plain
    for name, (t, acc) in plain_items:
        sz  = size_of(name)
        col = baseline_shades.get(sz, "#aaaaaa")
        ax.scatter(t, acc, s=150, marker="o", color=col, label=name)

    # stars   – Split Reasoner
    for name, (t, acc) in spec_items:
        sz  = size_of(name)
        col = spec_shades.get(sz, "#888888")
        label = name.replace("SpecR", "SplitReasoner")
        ax.scatter(t, acc, s=240, marker="*", color=col, label=label)

    # dashed trend‑lines
    plain_xy = np.array([p[1] for p in plain_items])
    spec_xy  = np.array([p[1] for p in spec_items])

    ax.plot(plain_xy[:, 0], plain_xy[:, 1], linestyle="--", color="grey", linewidth=1)
    ax.plot(spec_xy[:, 0],  spec_xy[:, 1],  linestyle="--", color="grey", linewidth=1)

    # cosmetics
    ax.set_xlabel("Mean evaluation time (seconds)", fontsize=16)
    ax.set_ylabel("Mean AIME24 Accuracy",          fontsize=16)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # ordered legend
    desired_order = ["1.5B", "8B", "14B", "32B",
                     "SplitReasoner 8B",
                     "SplitReasoner 14B",
                     "SplitReasoner 32B"]

    handles, labels = ax.get_legend_handles_labels()
    ordering = sorted(
        range(len(labels)),
        key=lambda i: desired_order.index(labels[i])
    )
    ax.legend([handles[i] for i in ordering],
              [labels[i]  for i in ordering],
              fontsize=11, framealpha=0.95, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig("accuracy_to_latency_teaser.pdf", format="pdf")
    print("Saved → accuracy_to_latency_teaser.pdf")


graph_type_1()
graph_type_2()
graph_type_3()
graph_teaser()