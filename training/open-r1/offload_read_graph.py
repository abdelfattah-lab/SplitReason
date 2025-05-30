import pickle
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
    # 1) Load the pickle file
    with open("bigmodel_span_text.pkl", "rb") as f:
        outputs = pickle.load(f)
    # 2) Create a 5x2 figure
    fig, axs = plt.subplots(5, 2, figsize=(14, 10))
    axs = axs.flatten()

    # 3) For each of the 10 items in 'outputs', compute mask and plot
    for i in range(10):
        ax = axs[i]
        text = outputs[i]

        mask = get_bigmodel_mask(text)

        # If the text is empty or mask is empty, skip
        if not text or not mask:
            ax.set_title(f"Example {i} (no content)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # x from 0 to 1 across the character range
        x = [k / len(mask) for k in range(len(mask))]
        y = mask  # 0/1 values
        # calculate coverage (percentage of 1s)
        coverage = 100.0 * sum(mask) / len(mask)

        # Plot with step
        ax.step(x, y, where='post')
        ax.set_ylim(-0.1, 1.1)  # 0 or 1 only
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"Example {i} ({coverage:.1f}% covered)")

    plt.tight_layout()
    plt.savefig("switch_behavior.pdf")
    plt.clf()   # Clear the figure from memory
    plt.close() # Close the plotting window

if __name__ == "__main__":
    main()
