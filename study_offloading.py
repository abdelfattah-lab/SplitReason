#!/usr/bin/env python3
"""
Analyze <bigmodel> tag coverage in response texts, handle multiple responses per
example, and compute pass@1 from the `exact_matches` vectors.

Changes versus the original script
----------------------------------
* Iterate over **every** response in each example, building a *list‑of‑lists*
  (`coverage_per_example`) so we can later examine variability within an
  example ( std‑dev across responses) as well as global statistics.
* Collect **all** `exact_matches` vectors and report `pass@1` as their overall
  mean.
* Print – for every processed *.jsonl* file –
    * mean / std / median / min / max coverage **across all responses**
    * average *within‑example* coverage std‑dev
    * pass@1
* Existing 5×6 PDF grid is kept (still showing the *first* response of up to
  30 examples) so earlier visual behaviour is preserved.

Run the script as usual; it will save one `*_coverage.pdf` next to each input
json file and print the new statistics to stdout.
"""

import json
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

summary_rows: List[List[str]] = []         # will collect one row per run

def get_bigmodel_mask(text: str, open_tag: str = "<bigmodel>", close_tag: str = "</bigmodel>") -> List[int]:
    """Return a 0/1 mask marking characters inside <bigmodel>…</bigmodel>."""

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


def coverage_for_text(text: str) -> float:
    """Return percentage of characters wrapped in <bigmodel> tags (NaN if empty)."""

    mask = get_bigmodel_mask(text)
    if not mask:
        return float("nan")
    return 100.0 * sum(mask) / len(mask)


def main() -> None:
    # ---------------------------------------------------------------------
    # Adjust this path as needed
    # json_folder_dict = {
    #     "GRPO_15B_12Aug": "/home/ya255/projects/SpeculativeReasoning/log_traces/GRPO_15B_12Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "ORIG_7B_12Aug": "/home/ya255/projects/SpeculativeReasoning/log_traces/ORIG_7B_12Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "ORIG_15B_12Aug": "/home/ya255/projects/SpeculativeReasoning/log_traces/ORIG_15B_12Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR8B": "/home/ya255/projects/SpeculativeReasoning/amazing_log_traces/SPECR_8B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR14B": "/home/ya255/projects/SpeculativeReasoning/amazing_log_traces/SPECR_14B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR32B": "/home/ya255/projects/SpeculativeReasoning/amazing_log_traces/SPECR_32B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    # }
    # json_folder_dict = {
    #     "Repeat8_ONLY_8B": "/home/ya255/projects/SpeculativeReasoning/log_traces/ONLY_8B_13Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_ONLY_15B": "/home/ya255/projects/SpeculativeReasoning/log_traces/ONLY_15B_13Aug/meta-llama__Llama-2-7b-chat-hf",
    #     # "ORIG_15B_12Aug": "/home/ya255/projects/SpeculativeReasoning/log_traces/ORIG_15B_12Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR8B": "/home/ya255/projects/SpeculativeReasoning/log_traces/SPECR_8B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR14B": "/home/ya255/projects/SpeculativeReasoning/log_traces/SPECR_14B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    #     "Repeat8_SPECR32B": "/home/ya255/projects/SpeculativeReasoning/log_traces/SPECR_32B_10Aug/meta-llama__Llama-2-7b-chat-hf",
    # }
    base_path = "/mnt/home/ya255/projects/SplitReason/log_traces/"
    json_folder_dict = {
        "E2EGRPO": "/mnt/home/ya255/projects/SplitReason/log_traces/NewModel_e2egrpo_8b_v2/meta-llama__Llama-2-7b-chat-hf"
        # # vllm-spec service for accuracy measurement with 1.5B model
        # "vlspec_ONLY_15B": "/home/ya255/projects/SpeculativeReasoning/log_traces/vlspec_ONLY_15B/meta-llama__Llama-2-7b-chat-hf",
        # # vllm-spec service with no-prefix-caching for accuracy measurement with 1.5B model
        # "vlspec_ONLY_15B_no_prefix": "/home/ya255/projects/SpeculativeReasoning/log_traces/vlspec_ONLY_15B_no_prefix/meta-llama__Llama-2-7b-chat-hf",
        # # vllm-spec service with no prefix caching and no chunked prefill for accuracy measurement with 1.5B model
        # "vlspec_ONLY_15B_no_prefix_no_chunk": "/home/ya255/projects/SpeculativeReasoning/log_traces/vlspec_ONLY_15B_no_prefix_no_chunk/meta-llama__Llama-2-7b-chat-hf",
        # # vllm base service for accuracy measurement with 1.5B model
        # "ONLY_15B_vllm": "/home/ya255/projects/SpeculativeReasoning/log_traces/vLLM_ONLY_15B_13Aug/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        # # vllm base service with \boxed prompt for accuracy measurement with 1.5B model
        # "ONLY_15B_vllm_fixp": "/home/ya255/projects/SpeculativeReasoning/log_traces/vLLM_ONLY_15B_13Aug_FixPrompt/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
        
    }
    # list all folders in /home/ya255/projects/SpeculativeReasoning/log_traces/
    fol_path_list = os.listdir(base_path)
    # use name of folder as key and path as value
    json_folder_dict = {f: os.path.join(base_path, f) for f in fol_path_list}
    # list the folder inside each value, add that to the path
    for k, v in json_folder_dict.items():
        # list all folders in v
        sub_fol_path_list = os.listdir(v)
        # use name of folder as key and path as value
        json_folder_dict[k] = os.path.join(v, sub_fol_path_list[0])
    # ---------------------------------------------------------------------
    for run_name, json_folder in json_folder_dict.items():
        if not os.path.isdir(json_folder):
            json_files = [json_folder]
        else:
            json_files = [
                os.path.join(json_folder, f)
                for f in os.listdir(json_folder)
                if f.startswith("samples_") and f.endswith(".jsonl") and "aime24" in f
            ]

        for json_file in json_files:
            with open(json_file, "r") as f:
                data = [json.loads(line) for line in f]

            # Collect per‑example coverage lists and exact‑matches
            coverage_per_example: List[List[float]] = []
            all_exact_matches: List[int] = []

            # Prepare figure (still visualising only the first response per example)
            fig, axs = plt.subplots(5, 6, figsize=(18, 10))
            axs = axs.flatten()

            for i, item in enumerate(data):
                # -- coverage for *every* response --------------------------------
                cov_this_example: List[float] = []
                for resp in item.get("resps", []):
                    # `resp` may itself be a list (e.g. beam search) or a bare str
                    text = resp[0] if isinstance(resp, list) else resp
                    if text:
                        cov_this_example.append(coverage_for_text(text))
                if cov_this_example:
                    coverage_per_example.append(cov_this_example)

                # -- exact matches -------------------------------------------------
                all_exact_matches.extend(item.get("exact_matches", []))

                # -- plot only the first response for up to 30 examples ------------
                if i < 30 and item.get("resps"):
                    first_resp = (
                        item["resps"][0][0]
                        if isinstance(item["resps"][0], list)
                        else item["resps"][0]
                    )
                    ax = axs[i]
                    mask = get_bigmodel_mask(first_resp)
                    if mask:
                        x = np.linspace(0, 1, len(mask))
                        ax.step(x, mask, where="post")
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_xlim(0, 1)
                        ax.set_yticks([])
                        first_cov = 100.0 * sum(mask) / len(mask)
                        ax.set_title(f"Example {i} ({first_cov:.1f}% covered)")
                    else:
                        ax.set_title(f"Example {i} (empty mask)")
                        ax.set_xticks([])
                        ax.set_yticks([])

            # Save the PDF grid next to the input file
            plt.tight_layout()
            # make a new directory of 'coverage_plots'
            if not os.path.exists("coverage_plots"):
                os.makedirs("coverage_plots")
            # out_pdf = os.path.splitext(os.path.basename(json_file))[0] + "_coverage.pdf"
            # put pdf in coverage_plots
            out_pdf = os.path.join("coverage_plots", os.path.splitext(os.path.basename(json_file))[0] + "_coverage.pdf")
            plt.savefig(out_pdf)
            plt.close()

            # --------------------------- statistics ------------------------------
            flat_cov = [c for sub in coverage_per_example for c in sub]
            overall_mean = np.nanmean(flat_cov)
            overall_std = np.nanstd(flat_cov)
            overall_median = np.nanmedian(flat_cov)
            overall_min = np.nanmin(flat_cov)
            overall_max = np.nanmax(flat_cov)

            within_std = [np.nanstd(sub) for sub in coverage_per_example if len(sub) > 1]
            mean_within_std = np.nanmean(within_std) if within_std else float("nan")

            pass_at_1 = np.mean(all_exact_matches) if all_exact_matches else float("nan")

            # --------------------------- report ----------------------------------
            print("Experiment:", run_name)
            print(
                (
                    "Coverage – mean: {mean:.2f} %, std: {std:.2f} %, median: {median:.2f} %,\n"
                    "           min:  {min:.2f} %, max: {max:.2f} %"
                ).format(
                    mean=overall_mean,
                    std=overall_std,
                    median=overall_median,
                    min=overall_min,
                    max=overall_max,
                )
            )
            # print(f"Average within‑example coverage std‑dev: {mean_within_std:.2f} %")
            print(f"pass@1: {pass_at_1:.3f}\n")
            summary_rows.append(
                [
                    run_name,
                    f"{pass_at_1:.3f}",
                    f"{overall_mean:.2f}",
                    f"{overall_median:.2f}",
                ]
            )

    headers = ["Experiment", "pass@1", "Mean Cov (%)", "Median Cov (%)"]
    print("\n===== SUMMARY =====")
    if tabulate:
        print(tabulate(summary_rows, headers=headers, tablefmt="github"))
    else:
        col_w = [max(len(str(x)) for x in col) for col in zip(headers, *summary_rows)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        for row in summary_rows:
            print(fmt.format(*row))

if __name__ == "__main__":
    main()
