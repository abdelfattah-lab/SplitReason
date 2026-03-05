"""
SplitReason-specific wandb logging callback for veRL.

Logs metrics that are unique to the speculative reasoning setup:
- gate/mean_calls: average <bigmodel> count per completion
- gate/frac_balanced: fraction with balanced tags
- metrics/offload_frac_mean: fraction of response tokens with mask=0
- metrics/char_coverage_mean: char-level coverage ratio
- metrics/f_any_offload: fraction of completions with any offloading

Usage:
    This callback is meant to be integrated into veRL's training loop.
    Import and call log_splitreason_metrics() after each training step
    with the batch data.

    Since veRL doesn't have a simple callback hook like TRL's Trainer,
    this module provides utility functions that can be called from a
    custom training script or monkey-patched into the training loop.
"""

import re
from typing import Optional

import numpy as np


def compute_splitreason_metrics(
    completions: list[str],
    response_masks: Optional[list[list[int]]] = None,
) -> dict[str, float]:
    """
    Compute SplitReason-specific metrics from a batch of completions.

    Args:
        completions: List of decoded completion strings
        response_masks: Optional list of response mask lists (1=small, 0=big)

    Returns:
        Dict of metric_name -> value for logging
    """
    if not completions:
        return {}

    n = len(completions)
    metrics = {}

    # gate/mean_calls: average <bigmodel> count per completion
    bigmodel_counts = [text.count("<bigmodel>") for text in completions]
    metrics["gate/mean_calls"] = np.mean(bigmodel_counts)

    # gate/frac_balanced: fraction with balanced open/close tags
    balanced = [
        int(text.count("<bigmodel>") == text.count("</bigmodel>"))
        for text in completions
    ]
    metrics["gate/frac_balanced"] = np.mean(balanced)

    # metrics/f_any_offload: fraction with at least one <bigmodel> tag
    any_offload = [int(c > 0) for c in bigmodel_counts]
    metrics["metrics/f_any_offload"] = np.mean(any_offload)

    # metrics/char_coverage_mean: average char-level coverage ratio
    coverages = []
    for text in completions:
        total = len(text)
        if total == 0:
            coverages.append(0.0)
            continue
        segments = re.findall(r"<bigmodel>(.*?)</bigmodel>", text, re.DOTALL)
        bigmodel_chars = sum(len(seg) for seg in segments)
        coverages.append(bigmodel_chars / total)
    metrics["metrics/char_coverage_mean"] = np.mean(coverages)

    # metrics/offload_frac_mean: fraction of response tokens with mask=0
    if response_masks is not None:
        offload_fracs = []
        for mask in response_masks:
            total = len(mask)
            if total == 0:
                offload_fracs.append(0.0)
                continue
            big_tokens = sum(1 for m in mask if m == 0)
            offload_fracs.append(big_tokens / total)
        metrics["metrics/offload_frac_mean"] = np.mean(offload_fracs)

    return metrics


def log_splitreason_metrics_wandb(
    completions: list[str],
    response_masks: Optional[list[list[int]]] = None,
    step: Optional[int] = None,
):
    """
    Compute and log SplitReason metrics to wandb.

    Call this after each training step if wandb is available.
    """
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return

    metrics = compute_splitreason_metrics(completions, response_masks)
    if metrics:
        wandb.log(metrics, step=step)
