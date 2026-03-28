from __future__ import annotations

import numpy as np


def _normalize_channel(channel: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    lo = np.percentile(channel, low_pct)
    hi = np.percentile(channel, high_pct)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((channel - lo) / (hi - lo), 0.0, 1.0)


def _to_hwc(feature_map: np.ndarray, input_layout: str) -> np.ndarray:
    x = feature_map
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D feature map, got shape {x.shape}")

    layout = input_layout.upper()
    if layout == "HWC":
        return x
    if layout == "CHW":
        return np.transpose(x, (1, 2, 0))
    if layout != "AUTO":
        raise ValueError(f"Unsupported input_layout='{input_layout}'")

    # AUTO fallback:
    # If first dim is clearly dominant, this is likely CHW [C,H,W].
    if x.shape[0] > x.shape[1] and x.shape[0] > x.shape[2]:
        return np.transpose(x, (1, 2, 0))
    return x


def pca_rgb(feature_map: np.ndarray, input_layout: str = "AUTO") -> np.ndarray:
    """Convert one feature map to a PCA RGB visualization.

    Accepted shapes:
      - [H, W, C] (NHWC single map)
      - [C, H, W] (NCHW single map)
      - [1, H, W, C] or [1, C, H, W] (batch=1)
    Returns:
      - uint8 RGB image [H, W, 3]
    """
    x = _to_hwc(feature_map, input_layout=input_layout)

    h, w, c = x.shape
    if c < 3:
        raise ValueError(f"Need at least 3 channels for PCA visualization, got C={c}")

    flat = x.astype(np.float32).reshape(-1, c)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vt[:3].T

    rgb = np.zeros_like(proj, dtype=np.float32)
    for k in range(3):
        rgb[:, k] = _normalize_channel(proj[:, k])

    return (rgb.reshape(h, w, 3) * 255.0).astype(np.uint8)
