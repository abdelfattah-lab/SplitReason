import matplotlib.pyplot as plt
from datasets import load_dataset

# Load the dataset
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")
train_ds = ds['train']

# Collect data
generation_lengths = []
generation_ids = []
from tqdm import tqdm
gen_id = 0
for item in tqdm(train_ds):
    for gen in item['generations']:
        generation_lengths.append(len(gen))
        generation_ids.append(gen_id)
        gen_id += 1

# Plot: Generation ID vs. String Length
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(generation_ids, generation_lengths, linewidth=0.5)
plt.xlabel("Generation ID")
plt.ylabel("String Length")
plt.title("Generation ID vs. String Length")
plt.tight_layout()
plt.savefig("genid_vs_strlen.png")
plt.close()

# Plot: Histogram of String Lengths (bin size 500)
plt.figure(figsize=(10, 6), dpi=600)
bins = range(0, max(generation_lengths) + 500, 500)
plt.hist(generation_lengths, bins=bins, edgecolor='black')
plt.xlabel("String Length")
plt.ylabel("Frequency")
plt.title("Histogram of Generation String Lengths")
plt.tight_layout()
plt.savefig("strlen_hist.png")
plt.close()

print("Plots saved as 'genid_vs_strlen.png' and 'strlen_hist.png'")
