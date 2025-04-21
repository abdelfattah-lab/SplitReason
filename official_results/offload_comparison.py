
import random
import pickle
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk


# ------------------------- helpers ------------------------- #
def get_bigmodel_mask(
    text: str,
    open_tag: str = "<bigmodel>",
    close_tag: str = "</bigmodel>",
) -> list[int]:
    """Return a 0/1 mask (one entry per character) for the <bigmodel> regions."""
    mask = [0] * len(text)
    start_index = 0
    while True:
        open_pos = text.find(open_tag, start_index)
        if open_pos == -1:
            break
        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:  # no close‑tag: mark to the end
            mask[open_pos:] = [1] * (len(text) - open_pos)
            break
        region_end = close_pos + len(close_tag)
        mask[open_pos:region_end] = [1] * (region_end - open_pos)
        start_index = region_end
    return mask


# ------------------------- data ------------------------- #
# 1) Annotated DeepSeek dataset (from disk if it exists, otherwise Hub)
# try:
    # ds = load_from_disk("OpenR1_Math_Annotated_DeepSeek")
# except Exception:
ds = load_dataset("akhauriyash/OpenR1_Math_SpeculativeReasoning")

deepseek_msgs = ds["train"]["messages"]        # list of dialogue turns
deepseek_pool = list(range(len(deepseek_msgs)))

# 2) Model generations stored as pickle files
with open("bigmodel_span_text_sft.pkl", "rb") as f:
    sft_outputs: dict[int, str] = pickle.load(f)

with open("bigmodel_span_text_grpo.pkl", "rb") as f:
    grpo_outputs: dict[int, str] = pickle.load(f)

# SFT
# Average coverage: 4.66%          std: 4.45%      min: 0.00%      max: 12.07%
# GRPO
# Average coverage: 17.18%         std: 9.47%      min: 1.65%      max: 27.98%



# ------------------------- sampling ------------------------- #
random.seed(42)                                 # reproducibility
n_samples = 10

sample_ids_dataset = random.sample(deepseek_pool, n_samples)
sample_ids_sft     = random.sample(list(sft_outputs.keys()), n_samples)
sample_ids_grpo    = random.sample(list(grpo_outputs.keys()), n_samples)

plots = [
    [deepseek_msgs[i][1]["content"] for i in sample_ids_dataset],
    [sft_outputs[i]                  for i in sample_ids_sft],
    [grpo_outputs[i]                 for i in sample_ids_grpo],
]
titles = ["Dataset", "Supervised Fine‑Tuned", "GRPO"]


# ------------------------- plotting ------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)
for ax, title, texts in zip(axes, titles, plots):
    for row, text in enumerate(texts):
        mask = get_bigmodel_mask(text)
        if any(mask):                                 # skip if no <bigmodel> tag
            x = [i / len(mask) for i in range(len(mask))]
            y = [m + row * 1.1 for m in mask]        # vertical offset per sample
            ax.step(x, y, where="post")

    ax.set_title(title, fontsize=22)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, n_samples * 1.1)
    ax.set_yticks([])
    # Set x-ticks font size 18
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlabel("Normalized String Length", fontsize=22)

axes[0].set_ylabel("Offload", fontsize=22)
plt.tight_layout()
plt.savefig("bigmodel_offload_comparison.pdf")
plt.show()