"""SPIDER instance labels."""
from __future__ import annotations

import numpy as np


def semantic_from_instance(instance: np.ndarray) -> np.ndarray:
    """Return background, vertebra, disc and canal as 0, 1, 2, 3."""
    source = np.asarray(instance)
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("Instance labels must be integers")
    vertebra = (source >= 1) & (source <= 25)
    disc = (source >= 201) & (source <= 225)
    allowed = (source == 0) | vertebra | disc | (source == 100)
    if not np.all(allowed):
        raise ValueError(f"Unknown instance labels: {np.unique(source[~allowed]).tolist()}")
    semantic = np.zeros(source.shape, dtype=np.uint8)
    semantic[vertebra] = 1
    semantic[disc] = 2
    semantic[source == 100] = 3
    return semantic
