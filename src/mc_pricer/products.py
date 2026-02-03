from __future__ import annotations

import numpy as np


def payoff_call(ST: np.ndarray, K: float) -> np.ndarray:
    """Vectorized payoff for a European call: max(ST - K, 0)."""
    if K <= 0.0:
        raise ValueError("K must be > 0")
    return np.maximum(ST - K, 0.0)


def payoff_put(ST: np.ndarray, K: float) -> np.ndarray:
    """Vectorized payoff for a European put: max(K - ST, 0)."""
    if K <= 0.0:
        raise ValueError("K must be > 0")
    return np.maximum(K - ST, 0.0)
