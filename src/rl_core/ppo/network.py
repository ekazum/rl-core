"""``PPOPolicyNetwork`` — actor-critic as two separate SiLU stacks.

Independent policy and value networks, not a shared trunk (stable-baselines3's
``net_arch=dict(pi=..., vf=...)``). Emits ``(B, A + 1)``: action logits then
the value, so one forward serves both.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Logit floor for invalid actions; finite so softmax never yields NaN.
MASKED_LOGIT = -1e9

# Default width of each stack.
HIDDEN = (64, 64)


class PPOPolicyNetwork(nn.Module):
    """Independent policy and value stacks over the same observation."""

    HIDDEN: tuple[int, ...] = HIDDEN

    def __init__(self, input_dim: int, num_actions: int,
                 hidden: tuple = HIDDEN) -> None:
        """Build both stacks; no dropout, so a stored log-prob is reproducible."""
        super().__init__()
        if input_dim < 1 or num_actions < 1:
            raise ValueError(
                f"input_dim and num_actions must be >= 1; got "
                f"{input_dim}, {num_actions}")
        self._hidden = tuple(int(h) for h in hidden)
        self._policy = self._stack(input_dim, self._hidden, num_actions)
        self._value = self._stack(input_dim, self._hidden, 1)
        logger.info(
            "PPOPolicyNetwork: separate %s SiLU stacks, %d actions, "
            "%d parameters.", list(self._hidden), num_actions,
            sum(p.numel() for p in self.parameters()))

    @staticmethod
    def _stack(input_dim: int, hidden: tuple, out_dim: int) -> nn.Sequential:
        """``Linear -> SiLU`` per hidden width, then a linear output."""
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.SiLU())
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    @property
    def hidden(self) -> tuple:
        """Width of each stack, for the checkpoint's meta."""
        return self._hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, A + 1)``: action logits in ``[:, :-1]``, the value in ``[:, -1:]``."""
        return torch.cat([self._policy(x), self._value(x)], dim=1)

    @staticmethod
    def split(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate a forward's output into ``(logits (B, A), value (B,))``."""
        return out[:, :-1], out[:, -1]

    @staticmethod
    def masked_log_probs(logits: torch.Tensor,
                         invalid: torch.Tensor) -> torch.Tensor:
        """Log-softmax over the valid actions; ``invalid`` is True where masked."""
        return torch.log_softmax(logits.masked_fill(invalid, MASKED_LOGIT), dim=1)

    @staticmethod
    def entropy(log_probs: torch.Tensor,
                invalid: torch.Tensor) -> torch.Tensor:
        """``(B,)`` entropy of each row's masked action distribution."""
        probs = log_probs.exp()
        return -(probs * log_probs.masked_fill(invalid, 0.0)).sum(dim=1)
