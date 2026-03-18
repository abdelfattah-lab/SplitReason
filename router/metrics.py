"""
Router-specific wandb logging metrics.

Metrics:
- router/offload_frac_mean: fraction of response tokens from big model
- router/f_any_offload: fraction of completions with any offloading
- router/mean_yields: average yield actions per sequence
- router/mean_switches: average S↔L transitions per sequence
"""

from typing import Optional

import numpy as np


def compute_router_metrics(
    completions: list[str],
    response_masks: Optional[list[list[int]]] = None,
    routing_data: Optional[dict] = None,
) -> dict[str, float]:
    """Compute router-specific metrics from a batch of completions."""
    if not completions:
        return {}

    metrics = {}

    if response_masks is not None:
        offload_fracs = [
            sum(1 for m in mask if m == 0) / max(len(mask), 1)
            for mask in response_masks
        ]
        metrics["router/offload_frac_mean"] = float(np.mean(offload_fracs))
        metrics["router/f_any_offload"] = float(np.mean([f > 0 for f in offload_fracs]))

    if routing_data is not None:
        if "num_yields" in routing_data:
            metrics["router/mean_yields"] = float(np.mean(routing_data["num_yields"]))
        if "num_switches" in routing_data:
            metrics["router/mean_switches"] = float(np.mean(routing_data["num_switches"]))

    return metrics
