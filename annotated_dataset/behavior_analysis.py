import matplotlib.pyplot as plt
from datasets import load_from_disk
import random

# 1) Load your two annotated datasets from disk
ds_deepseek = load_from_disk("OpenR1_Math_Annotated_DeepSeek")
# ds_gpt = load_from_disk("OpenR1_Math_Annotated_GPT4o")

# 2) A helper function to build a [0,1] mask showing where <bigmodel> ... <\bigmodel> is active
def get_bigmodel_mask(text, open_tag="<bigmodel>", close_tag="<\\bigmodel>"):
    """
    Return a list of 0/1 (len == len(text)) indicating 
    which character positions are inside <bigmodel> ... <\bigmodel>.
    """
    mask = [0] * len(text)
    start_index = 0
    
    while True:
        # Find next opening tag
        open_pos = text.find(open_tag, start_index)
        if open_pos == -1:
            # No more occurrences
            break
        
        # Find the corresponding closing tag
        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:
            # If we can't find a close tag, stop
            break
        
        # Mark the range [open_pos, close_pos + len(close_tag)) as 1
        region_end = min(close_pos + len(close_tag), len(text))
        for i in range(open_pos, region_end):
            mask[i] = 1
        
        # Move ahead, searching after the close tag
        start_index = region_end
    
    return mask

# Setup: 5 rows x 2 columns = 10 subplots
fig, axs = plt.subplots(5, 2, figsize=(14, 20))
axs = axs.flatten()

for subplot_idx in range(10):
    ax = axs[subplot_idx]
    for j in range(10):
        # i = subplot_idx * 10 + j
        # Generate random index between 0 and len(ds_deepseek)
        i = random.randint(0, len(ds_deepseek) - 1)
        deepseek_text = ds_deepseek[i]["annotated_generations"][0]
        mask = get_bigmodel_mask(deepseek_text)
        if not mask:
            continue
        x = [k / len(mask) for k in range(len(mask))]
        y = [v + j * 1.1 for v in mask]  # vertical offset
        ax.step(x, y, where='post')

    ax.set_ylim(-0.1, 10 * 1.1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Examples {subplot_idx * 10} to {subplot_idx * 10 + 9}")

plt.tight_layout()
plt.savefig("switch_behavior.pdf")
plt.cla()
plt.clf()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('annotation_statistics.csv')

# # Clean percentage strings and convert to float
# # df['offload_percentage'] = df['offload_percentage'].str.replace('%', '').astype(float)

# # First plot: offload_percentage vs qid
# plt.figure(figsize=(12, 6))
# for model in df['model'].unique():
#     subset = df[df['model'] == model]
#     plt.plot(subset['qid'], subset['offload_percentage'], marker='o', label=model)
# plt.xlabel('Question ID (qid)', fontsize=18)
# plt.ylabel('Offload Percentage (%)', fontsize=18)
# plt.title('Offload Percentage by Question ID for GPT4o and Dpsr1', fontsize=18)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('offload_percentage_by_qid.png')
# plt.show()
# plt.cla()
# plt.clf()

# # Second plot: offload_tokens_per_offload_chars vs qid
# plt.figure(figsize=(12, 6))
# for model in df['model'].unique():
#     subset = df[df['model'] == model]
#     plt.plot(subset['qid'], subset['offload_tokens_per_offload_chars'], marker='o', label=model)
# plt.xlabel('Question ID (qid)', fontsize=18)
# plt.ylabel('Offload Tokens per Offload Chars', fontsize=18)
# plt.title('Tokens per Offloaded Char by Question ID for GPT4o and Dpsr1', fontsize=18)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('tokens_per_offload_char_by_qid.png')
# plt.show()
