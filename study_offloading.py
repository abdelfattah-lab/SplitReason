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

# mask = get_bigmodel_mask(test_string)
# # get coverage
# coverage = 100.0 * sum(mask) / len(mask)
# print(f"Coverage: {coverage:.1f}%")


def main():
    # 1) Load the data (adjust filename as needed)

    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/SR_15b_only_v0/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-09T12-02-01.951099.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/SR_15b_only_v1/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-09T13-20-54.608577.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/SR_15b_only_v2/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-09T13-31-10.265245.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-09T21-38-40.981932.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_1/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-09T23-33-25.841688.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_1/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T08-09-20.612682.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_1_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T10-28-20.137619.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_2_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T11-28-43.780710.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_2_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T11-45-51.441609.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_2_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T12-05-33.361131.jsonl"
    # json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_2_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T12-41-03.279610.jsonl"
    json_file = "/home/ya255/projects/SpeculativeReasoning/log_traces/specreason_test_2_nogrpo/meta-llama__Llama-2-7b-chat-hf/samples_aime24_nofigures_2025-04-10T13-22-13.516233.jsonl"
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

        # Example: assume the text of interest is in data[i]['resps'][0][0]
        # Adjust indexing if your data is different
        text = item['resps'][0][0] if item['resps'] else ""

        # Skip if no text
        if not text:
            ax.set_title(f"Example {i} (no text)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 3) Compute the mask
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
    print(f"Average coverage: {sum(coverage_list) / len(coverage_list):.2f}% \t std: {np.std(coverage_list):.2f}% \t min: {min(coverage_list):.2f}%\t max: {max(coverage_list):.2f}%")

if __name__ == "__main__":
    main()
