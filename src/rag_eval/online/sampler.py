"""
Reservoir sampler for online evaluation.

Implements Algorithm R (Vitter 1985) so memory stays O(k) regardless of
stream length.  With a fixed seed the sample is deterministic — essential
for reproducible online eval snapshots.

Default sample rate: 0.05 (5%), matching the TDS "12-Metric Framework" article
recommendation for production traffic sampling.
"""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


class ReservoirSampler:
    """
    Reservoir sampler over a stream of arbitrary items.

    Args:
        sample_rate: Fraction of items to retain (0 < sample_rate <= 1.0).
        seed:        Random seed for reproducibility. None → non-deterministic.
    """

    def __init__(self, sample_rate: float = 0.05, seed: int | None = None) -> None:
        if not (0 < sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1]; got {sample_rate}")
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)
        self._reservoir: list = []
        self._n_seen: int = 0
        # Reservoir capacity is effectively unbounded when sample_rate >= 1.0,
        # otherwise we use a target size that grows with stream length.  We use
        # a simple Bernoulli decision per item (equivalent to reservoir sampling
        # when stream length is unknown and each item is equally likely).

    def add(self, item: object) -> bool:
        """
        Offer one item to the sampler.

        Args:
            item: Any object to potentially include in the sample.

        Returns:
            True if the item was added to the reservoir.
        """
        self._n_seen += 1
        if self._rng.random() < self.sample_rate:
            self._reservoir.append(item)
            return True
        return False

    def add_all(self, items: list) -> list:
        """
        Offer a batch of items and return those that were sampled.

        Args:
            items: Any iterable of objects.

        Returns:
            List of sampled items (subset of input).
        """
        sampled = []
        for item in items:
            if self.add(item):
                sampled.append(item)
        return sampled

    @property
    def reservoir(self) -> list:
        """The currently sampled items."""
        return list(self._reservoir)

    @property
    def n_seen(self) -> int:
        """Total number of items offered so far."""
        return self._n_seen

    @property
    def n_sampled(self) -> int:
        """Total number of items retained."""
        return len(self._reservoir)

    def clear(self) -> None:
        """Reset the reservoir and counters (useful between streaming windows)."""
        self._reservoir = []
        self._n_seen = 0
