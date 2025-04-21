import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, load_from_disk   # choose either

###############################################################################
# 1. Load and filter the dataset (same logic as your original script)
###############################################################################
# ds_raw = load_from_disk("OpenR1_Math_Annotated_DeepSeek")
ds_raw = load_dataset("akhauriyash/OpenR1_Math_SpeculativeReasoning", split="train")
# # Take a subset of 500
# ds_raw = ds_raw.select(range(2000))

OPEN_TAG, CLOSE_TAG = "<bigmodel>", "<\\bigmodel>"
TAG_RE = re.compile(rf"{re.escape(OPEN_TAG)}(.*?){re.escape(CLOSE_TAG)}", re.DOTALL)

def strip_tags(text: str) -> str:
    return text.replace(OPEN_TAG, "").replace(CLOSE_TAG, "")

def char_mask(text: str,
              open_tag: str = "<bigmodel>",
              close_tag: str = "<\\bigmodel>") -> np.ndarray:
    """
    Scan `text` once and build a mask where 1 means the character is
    enclosed by <bigmodel> … </bigmodel>.  The mask length equals
    len(text) minus the bytes that belong to the tag strings themselves.
    """
    mask = []
    inside = False
    i = 0
    L_open = len(open_tag)
    L_close = len(close_tag)

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

###############################################################################
# 2. Keep only generations without the error string
###############################################################################
examples = []
for ex in ds_raw:
    gens_ok = [g for g in ex["annotated_generations"]
               if "Error processing Due To" not in g]
    if gens_ok:
        ex["annotated_generations"] = gens_ok
        examples.append(ex)

###############################################################################
# 3. Build (a) normalised masks for profile, (b) per‑question off‑load fraction
###############################################################################
NUM_BINS = 200
sum_mask   = np.zeros(NUM_BINS, dtype=float)
sumsq_mask = np.zeros(NUM_BINS, dtype=float)
frac_per_q = []

gen_count = 0                    # <<< NEW – real generation counter

for ex in tqdm(examples, desc="Processing"):
    fracs = []
    for g in ex["annotated_generations"]:
        gen_count += 1           # <<< count every generation
        m = char_mask(g)
        fracs.append(m.mean())

        # resample mask to NUM_BINS
        orig_x = np.linspace(0, 1, len(m), endpoint=False)
        new_x  = np.linspace(0, 1, NUM_BINS, endpoint=False)
        resampled = np.interp(new_x, orig_x, m.astype(float))
        sum_mask   += resampled
        sumsq_mask += resampled ** 2

    frac_per_q.append(np.mean(fracs))

# --- use the correct denominator --------------------------------------------
mean_mask = sum_mask / gen_count
std_mask  = np.sqrt((sumsq_mask / gen_count) - mean_mask**2)
###############################################################################

###############################################################################
# 4. Plot: profile + histogram of off‑load fractions
###############################################################################
fig, (ax_prof, ax_hist) = plt.subplots(1, 2, figsize=(21, 5))

# Left panel – profile
x_pct = np.linspace(0, 100, NUM_BINS)
ax_prof.plot(x_pct, mean_mask, linewidth=2)
ax_prof.fill_between(
    x_pct,
    np.clip(mean_mask - std_mask, 0, 1),
    np.clip(mean_mask + std_mask, 0, 1),
    alpha=0.3,
)
ax_prof.set_xlabel("Relative position within generation (%)", fontsize=22)
ax_prof.set_ylabel("Fraction off‑loaded", fontsize=22)
ax_prof.set_title("Where <bigmodel> spans appear", fontsize=22)
# x and y axis ticks 18
ax_prof.tick_params(axis='x', labelsize=18)
ax_prof.tick_params(axis='y', labelsize=18)

# Right panel – histogram of per‑question fractions
bins = np.linspace(0, 1, 101)            # 20 equal bins between 0 and 1
ax_hist.hist(frac_per_q, bins=bins, edgecolor="black")
ax_hist.set_xlabel("Off‑loaded fraction per question", fontsize=22)
ax_hist.set_ylabel("Questions", fontsize=22)
ax_hist.set_title("Distribution of off‑loading", fontsize=22)
# x and y axis ticks 18
ax_hist.tick_params(axis='x', labelsize=18)
ax_hist.tick_params(axis='y', labelsize=18)

plt.tight_layout()
plt.savefig("dataset_analysis.pdf")
print("Saved → dataset_analysis.pdf")
