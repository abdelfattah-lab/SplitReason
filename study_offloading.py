# load json
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_bigmodel_mask(text, open_tag="<bigmodel>", close_tag="</bigmodel>"):
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

def main():
    # 1) Load the data (adjust filename as needed)
    # json_folder = "/home/ya255/projects/SpeculativeReasoning/log_traces/APR20_SPECR_FULLTRAINGRPO_8B/meta-llama__Llama-2-7b-chat-hf"
    # json_folder = "/home/ya255/projects/SpeculativeReasoning/log_traces/SPECR_8B/meta-llama__Llama-2-7b-chat-hf"
    json_folder = "/home/ya255/projects/SpeculativeReasoning/log_traces/SPECR_7B_Qwen/meta-llama__Llama-2-7b-chat-hf"
    # list all files with prefix samples_
    print(f"Investigating json files in {json_folder}")
    # Check if json_folder is a folder or file
    if not os.path.isdir(json_folder):
        print(f"{json_folder} is not a directory")
        json_files = [json_folder]
    else:
        json_files = [f for f in os.listdir(json_folder) if f.startswith("samples_") and f.endswith(".jsonl")]
    for json_file in json_files:
        json_file = os.path.join(json_folder, json_file)
        with open(json_file, "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]

        # Create a 5×6 figure (to accommodate 30 items)
        fig, axs = plt.subplots(5, 6, figsize=(18, 10))
        axs = axs.flatten()  # Flatten to iterate easily

        coverage_list = []
        # 2) Iterate over each data item (up to 30)
        for i, item in enumerate(data):
            if i >= 30:
                break  # just in case there are more than 30 items

            ax = axs[i]

            text = item['resps'][0][0] if item['resps'] else ""

            # Skip if no text
            if not text:
                ax.set_title(f"Example {i} (no text)")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if not os.path.exists("text_samples"):
                os.makedirs("text_samples")
            with open(f"text_samples/sample_{i}.txt", "w") as f:
                f.write(text)

            mask = get_bigmodel_mask(text)

            if len(mask) == 0:
                ax.set_title(f"Example {i} (empty mask)")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            # 4) Prepare x and y for step plot
            x = [k / len(mask) for k in range(len(mask))]
            y = mask  # 0/1 list

            # 5) Plot
            ax.step(x, y, where='post')
            ax.set_ylim(-0.1, 1.1)  # only 0 or 1
            ax.set_xlim(0, 1)
            ax.set_yticks([])

            # 6) Add percentage label to the title
            coverage = 100.0 * sum(mask) / len(mask)
            coverage_list.append(coverage)
            ax.set_title(f"Example {i} ({coverage:.1f}% covered)")

        # Layout and save
        plt.tight_layout()
        plt.savefig("aime24_behavior.pdf")
        plt.clf()
        plt.close()
        # print(sorted(coverage_list))
        print(coverage_list[:10])
        median_coverage = np.median(coverage_list)
        print(f"Median coverage: {median_coverage}\t Average coverage: {sum(coverage_list) / len(coverage_list):.2f}% \t std: {np.std(coverage_list):.2f}% \t min: {min(coverage_list):.2f}%\t max: {max(coverage_list):.2f}%")

if __name__ == "__main__":
    main()



