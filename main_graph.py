#!/usr/bin/env python3
"""
Scan every  log_traces/**/results_*.json   file and build
autocollate_results.csv  +  a ready-to-use Pandas DataFrame.
"""

from pathlib import Path
import json, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR   = Path("log_traces")          # change if you move the folder
OUT_CSV    = "autocollate_results.csv"

TASK_REMAP = {                           # long-name  → short prefix
    "aime24_nofigures_maj8cov8": "AIME24",
    "aime25_nofigures_maj8cov8": "AIME25",
}
METRICS    = [                           # keys of interest (strip “,all”)
    "exact_match", "cov@2", "cov@4", "cov@8",
    "maj@2",   "maj@4",  "maj@8",
]
# ───────────────────────────────────────────────────────────────────────────

def latest_result_file(exp_dir: Path) -> Path | None:
    """Pick the newest results_*.json in an experiment dir."""
    files = sorted(
        exp_dir.rglob("results_*.json"),           # ← recursive!
        key=lambda p: p.stat().st_mtime
    )
    return files[-1] if files else None


rows: list[dict] = []
for exp_dir in BASE_DIR.iterdir():
    if not exp_dir.is_dir():
        continue

    res_path = latest_result_file(exp_dir)
    if res_path is None:
        print(f"⚠  No results_*.json found in {exp_dir}")
        continue

    with open(res_path) as f:
        data = json.load(f)

    row: dict[str, float | str | None] = {
        "experiment": exp_dir.name,
        "total_evaluation_time_seconds": float(data.get("total_evaluation_time_seconds", "nan")),
    }

    for long_name, short in TASK_REMAP.items():
        blk = data["results"].get(long_name, {})
        for m in METRICS:
            row[f"{short}_{m}"] = blk.get(f"{m},all")

    rows.append(row)

# build & save the DataFrame
df = (
    pd.DataFrame(rows)
      .sort_values("experiment")
      .reset_index(drop=True)
)

df.to_csv(OUT_CSV, index=False)
print(f"✅  Saved →  {OUT_CSV}")
print(df)               # quick peek


# ─── optional: clean the experiment names a little ─────────────
df["exp_clean"] = (
    df["experiment"]
      .str.replace(r"_\d+\w+$", "", regex=True)   # drop “_10Aug”-like suffix
      .str.replace(r"^(ONLY|SPECR)_", "", regex=True)  # drop prefix for legend
)

df["label"] = df["experiment"].str.replace(r"_\d+[A-Za-z]+$", "", regex=True)


tasks   = ["AIME24", "AIME25"]
groups  = ["ONLY", "SPECR"]          # column layout
x_vals  = [2, 4, 8]                  # abscissa
markers = {"cov": "o", "maj": "^"}   # circle / triangle

fig, axes = plt.subplots(
    nrows=len(tasks), ncols=len(groups),
    figsize=(12, 9), sharey="row", sharex="col"
)

for row, task in enumerate(tasks):
    for col, grp in enumerate(groups):
        ax  = axes[row, col]
        sub = df[df["experiment"].str.startswith(grp)]

        for _, r in sub.iterrows():
            cov_vals = [r[f"{task}_cov@{n}"] for n in x_vals]
            maj_vals = [r[f"{task}_maj@{n}"] for n in x_vals]

            base = r["label"]                        # e.g.  ONLY_14B   or  SPECR_8B
            ax.plot(x_vals, cov_vals, marker="o",
                    linewidth=1.5, markersize=8, label=f"{base} cov")
            ax.plot(x_vals, maj_vals, marker="^", linestyle="--",
                    linewidth=1.5, markersize=8, label=f"{base} maj")

        # cosmetics
        ax.set_xticks(x_vals)
        ax.set_xlabel("k  in  @k", fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{task}  score", fontsize=12)
        ax.set_title(f"{grp} experiments", fontsize=14)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

        # tidy legends: one per subplot (remove if too busy)
        ax.legend(fontsize=8, framealpha=0.95)

fig.suptitle("Coverage (cons) vs Majority (maj) across k ∈ {2,4,8}", fontsize=16)
fig.tight_layout(rect=[0, 0.00, 1, 0.96])
fig.savefig("cons_maj_study.pdf", format="pdf")
print("✅  wrote  cons_maj_study.pdf")

# ─── maj@8  vs  execution-time  (scatter + sorted line) ───────────────────
tasks  = ["AIME24", "AIME25"]          # ← subplot order: left → right
groups = {"ONLY": "#1f77b4",           # colour palette
          "SPECR": "#ff7f0e"}

marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(11, 5), sharey=True
)

for ax, task in zip(axes, tasks):
    for g_name, colour in groups.items():
        sub = df[df["experiment"].str.startswith(g_name)].copy()

        # sort by exec-time to avoid jagged connecting lines
        sub = sub.sort_values("total_evaluation_time_seconds")

        xs = sub["total_evaluation_time_seconds"].values
        ys = sub[f"{task}_maj@8"].values
        # ys = sub[f"{task}_exact_match"].values

        # scatter – one marker per run
        for i, (x, y, lbl) in enumerate(zip(xs, ys, sub["label"])):
            m = marker_cycle[i % len(marker_cycle)]
            ax.scatter(x, y, marker=m, s=140, color=colour,
                       edgecolor="k", linewidth=0.5,
                       label=f"{lbl} ({g_name})" if ax is axes[0] else None)

        # connect the points (ONE line per group)
        ax.plot(xs, ys, color=colour, linewidth=1.8, label=None)
                # label=f"{g_name} line" if ax is axes[0] else None)

    ax.set_title(f"{task}", fontsize=14)
    ax.set_xlabel("Execution time (s)", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

axes[0].set_ylabel("maj@8", fontsize=12)

# place one consolidated legend under the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=4, fontsize=9, framealpha=0.95)

fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig.savefig("maj8_vs_time.pdf", format="pdf")
print("✅  wrote  maj8_vs_time.pdf")
