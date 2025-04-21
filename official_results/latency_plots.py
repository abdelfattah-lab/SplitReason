# latency_plots.py
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Load and tidy the data
# -------------------------------------------------
df = pd.read_csv("latency_data.csv")

# Keep only the 4 000‑token rows
df = df[df["Sequence Length"] == 4000]

# Model size (numeric, billions of params)
df["Model Size (B)"] = (
    df["Params (B)"].astype(str).str.replace("B", "", regex=False).astype(float)
)

# Split by phase
prefill = df[df["Phase"] == "Prefill"]
decode  = df[df["Phase"] == "Decode"]

# -------------------------------------------------
# 2. Build ratio (prefill ÷ decode)
# -------------------------------------------------
merged = pd.merge(
    prefill,
    decode,
    on=["Model", "Params (B)", "Sequence Length"],
    suffixes=("_prefill", "_decode"),
)
merged["Ratio"] = merged["Tok/sec_prefill"] / merged["Tok/sec_decode"]

# -------------------------------------------------
# 3. Two‑column figure (14 × 5 inches)
# -------------------------------------------------
fig, (ax_comb, ax_ratio) = plt.subplots(1, 2, figsize=(21, 5))

# ---- Combined Prefill & Decode throughput ----
ax_decode = ax_comb.twinx()  # right‑hand y‑axis for decode

prefill_scatter = ax_comb.plot(
    prefill["Model Size (B)"],
    prefill["Tok/sec"],
    color="0",
    marker='o',
    # size of the marker
    markersize=10,
    label="Prefill",
)
decode_scatter = ax_decode.plot(
    decode["Model Size (B)"],
    decode["Tok/sec"],
    color="tab:red",
    marker='x',
    markersize=10,
    label="Decode",
)

ax_comb.set_title("Throughput vs. Model Size (4k tokens)", fontsize=22)
ax_comb.set_xlabel("Model size (B parameters)", fontsize=22)
ax_comb.set_ylabel("Prefill tokens / second", color="0", fontsize=22)
ax_decode.set_ylabel("Decode tokens / second", color="tab:red", fontsize=22)

ax_comb.tick_params(axis="both", which="major", labelsize=18, colors="0")
ax_decode.tick_params(axis="y", which="major", labelsize=18, colors="tab:red")
ax_comb.tick_params(axis="x", which="major", labelsize=18)

# Single legend
handles = [prefill_scatter, decode_scatter]
labels  = ["Prefill", "Decode"]

# ---- Ratio plot ----
ax_ratio.scatter(
    merged["Model Size (B)_prefill"],
    merged["Ratio"],
    color="0",
    marker='o',
    s=200,
)
ax_ratio.axhline(1.0, linestyle="--", color="gray")
ax_ratio.set_title("Prefill / Decode throughput ratio", fontsize=22)
ax_ratio.set_xlabel("Model size (B parameters)", fontsize=22)
ax_ratio.set_ylabel("Ratio", fontsize=22)
ax_ratio.tick_params(axis="both", which="major", labelsize=18)

fig.tight_layout()
fig.savefig("latency_plots.pdf", bbox_inches="tight")
