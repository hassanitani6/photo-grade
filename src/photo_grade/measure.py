"""Colour-region rules (sand, sky, skin) and objective region statistics of a photo."""
import numpy as np

from .grade import LUMA

MIN_SAND_FRACTION = 0.05

# name: (hue range deg, sat range, value range, rows (fraction of height), cols (fraction of width))
REGIONS = {
    "sand": ((15, 55), (0.15, 0.85), (0.12, 0.95), (0.25, 1.0), (0.0, 1.0)),
    "sky": ((180, 250), (0.15, 1.0), (0.0, 1.0), (0.0, 0.45), (0.0, 1.0)),
    "skin": ((5, 35), (0.2, 0.6), (0.25, 0.9), (0.2, 0.8), (0.2, 0.8)),
    # Overcast, hazy or sunset-bleached sky: bright and nearly colourless, so the blue-hue mask misses it.
    "sky_grey": ((0, 360), (0.0, 0.15), (0.6, 1.0), (0.0, 0.45), (0.0, 1.0)),
}
SKIN_BOX = (0.2, 0.8, 0.2, 0.8)  # rows then cols, fractions
SKIN_PROMINENT = 0.15  # above this skin fraction the centre box is assumed to be a person, not sand


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60
    h = np.where(d == 0, 0, h)
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return np.stack([h, s, mx], -1)


def _mask(hsv: np.ndarray, spec: tuple) -> tuple[np.ndarray, tuple[slice, slice]]:
    (h0, h1), (s0, s1), (v0, v1), (r0, r1), (c0, c1) = spec
    H, W = hsv.shape[:2]
    win = (slice(int(H * r0), int(H * r1)), slice(int(W * c0), int(W * c1)))
    h, s, v = hsv[win][..., 0], hsv[win][..., 1], hsv[win][..., 2]
    return (h >= h0) & (h <= h1) & (s >= s0) & (s <= s1) & (v >= v0) & (v <= v1), win


def _summary(hsv: np.ndarray, m: np.ndarray, win: tuple[slice, slice]) -> dict:
    sub = hsv[win]
    n = int(m.sum())
    out = {"frac": float(n / m.size), "v": None, "s": None, "h": None}
    if n:
        out.update(v=float(np.median(sub[..., 2][m])), s=float(np.median(sub[..., 1][m])), h=float(np.median(sub[..., 0][m])))
    return out


def region_stats(rgb: np.ndarray) -> dict:
    """rgb: HxWx3 floats 0..1, gamma-encoded as displayed. Fractions are of the pixels in each region's search area;
    V/S/H are medians. When skin is prominent in the centre box that box is left out of the sand mask."""
    hsv = to_hsv(rgb)
    luma = rgb @ LUMA
    masks = {name: _mask(hsv, spec) for name, spec in REGIONS.items()}
    stats = {}
    skin_m, skin_win = masks["skin"]
    skin_frac = skin_m.sum() / skin_m.size
    for name, (m, win) in masks.items():
        if name == "sand" and skin_frac >= SKIN_PROMINENT:
            H, W = hsv.shape[:2]
            r0, r1, c0, c1 = SKIN_BOX
            m = m.copy()
            rows, cols = win
            m[max(int(H * r0) - rows.start, 0):max(int(H * r1) - rows.start, 0),
              max(int(W * c0) - cols.start, 0):max(int(W * c1) - cols.start, 0)] = False
        stats[name] = _summary(hsv, m, win)
    stats["skin"]["approximate"] = True
    stats["luma"] = {"p2": float(np.percentile(luma, 2)), "mean": float(luma.mean()), "p98": float(np.percentile(luma, 98))}
    return stats
