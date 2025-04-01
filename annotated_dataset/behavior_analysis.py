import matplotlib.pyplot as plt
from datasets import load_from_disk

# 1) Load your two annotated datasets from disk
ds_deepseek = load_from_disk("OpenR1_Math_Annotated_DeepSeek")
ds_gpt = load_from_disk("OpenR1_Math_Annotated_GPT4o")

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

# 3) Prepare our figure with 5 rows, 2 columns => total 10 subplots
fig, axs = plt.subplots(5, 2, figsize=(12, 16))
axs = axs.flatten()  # so we can index them as axs[i] directly

# We'll plot the first 10 examples only
num_examples = 10
for i in range(num_examples):
    # 4) Extract the first annotated generation from each dataset
    #    (Assuming each example has at least one generation)
    deepseek_text = ds_deepseek[i]["annotated_generations"][0]
    gpt_text = ds_gpt[i]["annotated_generations"][0]
    
    # 5) Get the 0/1 masks
    deepseek_mask = get_bigmodel_mask(deepseek_text)
    gpt_mask = get_bigmodel_mask(gpt_text)
    
    # 6) Plot them on the same axes:
    ax = axs[i]
    
    # Create x-coordinates
    deepseek_x = range(len(deepseek_mask))
    gpt_x = range(len(gpt_mask))
    # Plot lines for each
    # We'll use a step-like plot so it looks clearer which parts are "on/off"
    ax.step(deepseek_x, [x+2 for x in deepseek_mask], where='post', color='red', label='DeepSeek')
    ax.step(gpt_x, gpt_mask, where='post', color='blue', label='GPT')
    
    # Some minor labeling:
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(0, max(len(deepseek_mask), len(gpt_mask)))
    ax.set_title(f"Example {i}")
    ax.legend()

# 7) Final layout and save
plt.tight_layout()
plt.savefig("switch_behavior.pdf")
plt.cla()
plt.clf()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('annotation_statistics.csv')

# Clean percentage strings and convert to float
# df['offload_percentage'] = df['offload_percentage'].str.replace('%', '').astype(float)

# First plot: offload_percentage vs qid
plt.figure(figsize=(12, 6))
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['qid'], subset['offload_percentage'], marker='o', label=model)
plt.xlabel('Question ID (qid)', fontsize=18)
plt.ylabel('Offload Percentage (%)', fontsize=18)
plt.title('Offload Percentage by Question ID for GPT4o and Dpsr1', fontsize=18)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('offload_percentage_by_qid.png')
plt.show()
plt.cla()
plt.clf()

# Second plot: offload_tokens_per_offload_chars vs qid
plt.figure(figsize=(12, 6))
for model in df['model'].unique():
    subset = df[df['model'] == model]
    plt.plot(subset['qid'], subset['offload_tokens_per_offload_chars'], marker='o', label=model)
plt.xlabel('Question ID (qid)', fontsize=18)
plt.ylabel('Offload Tokens per Offload Chars', fontsize=18)
plt.title('Tokens per Offloaded Char by Question ID for GPT4o and Dpsr1', fontsize=18)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tokens_per_offload_char_by_qid.png')
plt.show()
