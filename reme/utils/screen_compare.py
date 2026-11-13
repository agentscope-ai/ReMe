"""Bounded pixel comparison with overlapping tiles for small desktop updates."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ScreenDifference:
    """Ratios describe only pixels outside known assistant/clock regions."""

    changed: bool
    changed_ratio: float
    tile_changed_ratio: float


def compare_screens(
    previous,
    current,
    ignored_regions=(),
    *,
    long_edge: int = 1280,
    pixel_delta: int = 20,
    changed_ratio: float = 0.0005,
    tile_size: int = 32,
    tile_stride: int = 16,
    tile_changed_ratio: float = 0.015,
) -> ScreenDifference:
    """Preserve aspect ratio; either a global or local change warrants analysis."""
    from PIL import Image

    if previous.size != current.size:
        return ScreenDifference(True, 1.0, 1.0)
    scale = min(1.0, long_edge / max(current.size))
    width, height = (max(1, round(size * scale)) for size in current.size)
    a, b = (
        np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), dtype=np.int16)
        for image in (previous, current)
    )
    valid = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in ignored_regions:
        left, right = max(0, math.floor(x0 * scale)), min(width, math.ceil(x1 * scale))
        top, bottom = max(0, math.floor(y0 * scale)), min(height, math.ceil(y1 * scale))
        if right <= left or bottom <= top:
            continue
        # Padding also excludes resampling bleed at the region boundary.
        valid[max(0, top - 2) : min(height, bottom + 2), max(0, left - 2) : min(width, right + 2)] = False
    count = int(valid.sum())
    if count == 0:
        return ScreenDifference(True, 1.0, 1.0)
    differences = (np.abs(a - b).max(axis=2) > pixel_delta) & valid
    global_ratio = float(differences.sum() / count)
    block = min(tile_size, height, width)
    step = min(tile_stride, block)
    top = np.unique(np.append(np.arange(0, height - block + 1, step), height - block))[:, None]
    left = np.unique(np.append(np.arange(0, width - block + 1, step), width - block))[None, :]

    def sums(pixels):
        integral = np.pad(pixels.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (
            integral[top + block, left + block]
            - integral[top, left + block]
            - integral[top + block, left]
            + integral[top, left]
        )

    tile_counts, tile_valid = sums(differences), sums(valid)
    ratios = np.divide(tile_counts, tile_valid, out=np.zeros_like(tile_counts, dtype=float), where=tile_valid > 0)
    # Slivers along masks should not amplify a single changed pixel into a whole-tile change.
    ratios[tile_valid < block * block / 2] = 0
    local_ratio = float(ratios.max(initial=0))
    return ScreenDifference(
        global_ratio >= changed_ratio or local_ratio >= tile_changed_ratio,
        global_ratio,
        local_ratio,
    )
