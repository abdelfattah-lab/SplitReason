"""
RoutingHead: 2-layer MLP that makes binary yield/continue decisions
from the small model's hidden states at chunk boundaries.

~200K trainable parameters. The base LM stays frozen.
"""

import torch
import torch.nn as nn


class RoutingHead(nn.Module):
    """
    2-layer MLP: hidden_dim → intermediate_dim → 1

    forward(h) returns a logit; route_prob(h) returns sigmoid(logit).
    Action: 1 = yield to big model, 0 = continue with small model.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logit(s) — shape (..., 1)."""
        return self.fc2(self.act(self.fc1(h)))

    def route_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Returns p(yield) ∈ (0, 1) — shape (..., 1)."""
        return torch.sigmoid(self.forward(h))


def init_from_lm_head(router: RoutingHead, model: nn.Module) -> None:
    """
    Initialize router.fc1.weight from the 128 highest-variance rows of
    model.lm_head.weight. This gives the router a head-start by reusing
    the LM's output projection as feature extraction.
    """
    with torch.no_grad():
        lm_weight = model.lm_head.weight.data  # (vocab_size, hidden_dim)
        variances = lm_weight.var(dim=1)  # (vocab_size,)
        intermediate_dim = router.fc1.weight.size(0)
        _, top_indices = variances.topk(intermediate_dim)
        router.fc1.weight.copy_(lm_weight[top_indices])
        router.fc1.bias.zero_()


def save_router(router: RoutingHead, path: str) -> None:
    """Save router state_dict to disk."""
    torch.save(router.state_dict(), path)


def load_router(router: RoutingHead, path: str) -> None:
    """Load router state_dict from disk."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    router.load_state_dict(state_dict)
