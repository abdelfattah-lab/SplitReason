import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk

def get_bigmodel_mask(text, index, open_tag="<bigmodel>", close_tag="<\\bigmodel>"):
    mask = [0] * len(text)
    start_index = 0

    while True:
        open_pos = text.find(open_tag, start_index)
        if open_pos == -1:
            break  # no more openings

        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:
            # If we can't find a close tag, mark until the end of the text
            for i in range(open_pos, len(text)):
                mask[i] = 1
            break
        else:
            # Mark the region from <bigmodel> ... </bigmodel>
            region_end = close_pos + len(close_tag)
            for i in range(open_pos, region_end):
                mask[i] = 1
            start_index = region_end

    return mask
    
# def get_bigmodel_mask(text, index, open_tag="<bigmodel>", close_tag="<\\bigmodel>"):
#     """
#     Return a list of 0/1 (length == len(text)) indicating
#     which character positions are inside <bigmodel> ... <\bigmodel>.
#     """
#     mask = [0] * len(text)
#     start_index = 0
    
#     while True:
#         # Find next opening tag
#         open_pos = text.find(open_tag, start_index)
#         if open_pos == -1:
#             break
        
#         # Find the corresponding closing tag
#         close_pos = text.find(close_tag, open_pos + len(open_tag))
#         if close_pos == -1:
#             break
        
#         region_end = min(close_pos + len(close_tag), len(text))
#         for i in range(open_pos, region_end):
#             mask[i] = 1
        
#         start_index = region_end
#     # if sum(mask) > 0.8 * len(mask):
#         # import pdb; pdb.set_trace()
#     return mask

# 1) Load your annotated datasetfrom datasets import load_from_disk

ds_deepseek = load_from_disk("OpenR1_Math_Annotated_DeepSeek")

error_count = 0     # Number of generations containing the error string
total_gen_count = 0 # Total number of generations seen
empty_item_count = 0  # How many items become empty after filtering

filtered_dataset = []
for example in ds_deepseek:
    gens = example["annotated_generations"]
    
    # Count how many contain the error
    for gen in gens:
        if "Error processing Due To" in gen:
            error_count += 1
    
    # Filter out generations containing the error
    original_gens = example['generations']
    for og_, fg_ in zip(original_gens, gens):
        oglen_ = len(og_)
        fglen_ = len(fg_)
        oglens_ = (int(oglen_), int(oglen_*1.2))
        if fglen_ < oglens_[0] or fglen_ > oglens_[1]:
            print(f"Warning: generation length {fglen_} is not within 10% of original length {oglen_}")
            # import pdb; pdb.set_trace()
    gens_filtered = [g for g in gens if "Error processing Due To" not in g]
    
    # Update the example’s generations
    example["annotated_generations"] = gens_filtered
    
    # Keep track of total generations
    total_gen_count += len(gens)
    
    # Only keep this entire example if it has at least one valid generation left
    if len(gens_filtered) > 0:
        filtered_dataset.append(example)
    else:
        empty_item_count += 1

# Convert the filtered list back to a dataset if needed
# (e.g. if ds_deepseek is a HF dataset, you can do from_dataset dict, or just keep as a list)
ds_deepseek_filtered = filtered_dataset  # or re-wrap with Dataset.from_list(filtered_dataset)

# Print stats
pct_gen_removed = 100.0 * error_count / (total_gen_count if total_gen_count else 1)
pct_items_removed = 100.0 * empty_item_count / len(ds_deepseek)
print(f"Total generations: {total_gen_count}")
print(f"{pct_gen_removed:.2f}% of *generations* contained 'Error processing Due To' and were removed.")
print(f"{pct_items_removed:.2f}% of *items* became empty and were removed.")

##############################################################################
# (A) Compute the mean offloading fraction (and std dev) across normalized positions
##############################################################################

# Choose how many bins we want along the normalized string length [0, 1]
NUM_BINS = 100

# Arrays to accumulate sums for computing mean & variance
bin_sums = np.zeros(NUM_BINS, dtype=float)
bin_sumsq = np.zeros(NUM_BINS, dtype=float)
bin_counts = np.zeros(NUM_BINS, dtype=float)

# We'll also store offload percentage by example (for later plotting)
offload_percentages = []
qids = []
string_lengths = []

exp_track = 0
for i, example in enumerate(tqdm(ds_deepseek_filtered)):
    # Some datasets store the generation in a list; adjust if your data structure differs
    generations = example["annotated_generations"]
    if not generations:
        continue
    
    # text = generations[0]  # take the first generation
    for idxcurr, text in enumerate(generations):
        if idxcurr > 0:
            continue  # skip all but the first generation
        mask = get_bigmodel_mask(text, exp_track)
        
        length = len(text)
        if length == 0:
            continue  # skip if empty
        
        # (A1) Bin the offloading (0 or 1) across normalized positions
        for idx in range(length):
            # Normalized position in [0, 1)
            pos = float(idx) / length
            
            # Convert that to a bin index
            bin_idx = int(pos * NUM_BINS)
            # Make sure we don't go out of range if pos == 1.0
            if bin_idx == NUM_BINS:
                bin_idx = NUM_BINS - 1
            
            val = mask[idx]
            bin_sums[bin_idx] += val
            bin_sumsq[bin_idx] += val * val
            bin_counts[bin_idx] += 1
        
        # (A2) Compute overall offload percentage for this example
        offload_chars = sum(mask)
        offload_percentage = 100.0 * offload_chars / length
        offload_percentages.append(offload_percentage)
        string_lengths.append(length)
        
        # # If your dataset has a 'qid' or unique ID, store that. Otherwise, just store i.
        # if "qid" in example:
        #     qids.append(example["qid"])
        # else:
        qids.append(exp_track)
        exp_track += 1

# Compute mean and std dev for each bin
mean_bin = bin_sums / np.clip(bin_counts, 1e-9, None)
var_bin = (bin_sumsq / np.clip(bin_counts, 1e-9, None)) - (mean_bin ** 2)
std_bin = np.sqrt(np.maximum(var_bin, 0.0))

# Create an x-axis for plotting (normalized positions)
x_vals = np.linspace(0, 1, NUM_BINS)


# (C) Plot offload percentage by question ID
# paired = sorted(zip(qids, offload_percentages, string_lengths), key=lambda x: x[0])
paired = sorted(zip(qids, offload_percentages, string_lengths), key=lambda x: x[0])
sorted_qids, sorted_offloads, sorted_lengths = zip(*paired)

# kendall tau rank correlation of sorted_offloads and sorted_lengths
from scipy.stats import kendalltau
tau, p_value = kendalltau(sorted_offloads, sorted_lengths)
print(f"Kendall tau rank correlation: {tau:.4f}, p-value: {p_value:.4f}")
##############################################################################
# (B) Plot the average offloading fraction vs. normalized position
#     with shaded standard deviation.
##############################################################################
plt.figure(figsize=(10, 6))
plt.plot(x_vals, mean_bin, color='blue', label='Mean Offload Fraction')
plt.fill_between(
    x_vals,
    mean_bin - std_bin,
    mean_bin + std_bin,
    color='blue',
    alpha=0.2,
    label='Std Dev'
)
plt.xlabel("Normalized position in string", fontsize=14)
plt.ylabel("Offloading fraction (0 or 1)", fontsize=14)
plt.title("Average Offloading Behavior (DeepSeek)", fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("normalized_offloading_deepseek.png")
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Offload %
axs[0].plot(sorted_qids, sorted_offloads, marker='o', linestyle='-', color='green')
axs[0].set_ylabel("Offload Percentage (%)", fontsize=14)
axs[0].set_title("Offload Percentage by Question ID", fontsize=16)
axs[0].grid(True, alpha=0.3)

# Plot 2: String Length
axs[1].plot(sorted_qids, sorted_lengths, marker='o', linestyle='-', color='blue')
axs[1].set_xlabel("Question ID (qid)", fontsize=14)
axs[1].set_ylabel("String Length", fontsize=14)
axs[1].set_title("String Length by Question ID", fontsize=16)
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("offload_percentage_by_qid_deepseek.png")
plt.close()


# ##############################################################################
# # (C) Plot offload percentage by question ID (based on sum(mask) / total length)
# ##############################################################################
# plt.figure(figsize=(10, 6))

# # (C) Plot offload percentage by question ID
# # paired = sorted(zip(qids, offload_percentages, string_lengths), key=lambda x: x[0])
# paired = sorted(zip(qids, offload_percentages, string_lengths), key=lambda x: x[0])
# sorted_qids, sorted_offloads, sorted_lengths = zip(*paired)

# # kendall tau rank correlation of sorted_offloads and sorted_lengths
# from scipy.stats import kendalltau
# tau, p_value = kendalltau(sorted_offloads, sorted_lengths)
# print(f"Kendall tau rank correlation: {tau:.4f}, p-value: {p_value:.4f}")

# plt.figure(figsize=(10, 6))
# plt.plot(sorted_qids, sorted_offloads, marker='o', linestyle='-', color='green', label='Offload %')

# ax1 = plt.gca()  # current Axes
# ax2 = ax1.twinx()  # create a twin Axes sharing x-axis
# ax2.set_ylabel("String Length", fontsize=14)
# ax2.plot(sorted_qids, sorted_lengths, marker='o', linestyle='-', color='blue', label='String Length')

# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

# plt.xlabel("Question ID (qid)", fontsize=14)
# # plt.ylabel("Offload Percentage (%)", fontsize=14)
# ax1.set_ylabel("Offload Percentage (%)", fontsize=14)
# ax2.set_ylabel("String Length", fontsize=14)
# plt.title("Offload Percentage by Question ID (DeepSeek)", fontsize=16)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("offload_percentage_by_qid_deepseek.png")
# plt.close()