import numpy as np


def make_bev(pts: np.ndarray,
             x_range: tuple = (0, 70),
             y_range: tuple = (-40, 40),
             resolution: float = 0.1) -> np.ndarray:
    """Render a bird's-eye view height/density image from velodyne points.

    Args:
        pts: (N, 4) array x, y, z, intensity
        x_range: forward range in metres
        y_range: lateral range in metres (left/right)
        resolution: metres per pixel

    Returns:
        RGB image as uint8 (H, W, 3)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    W = int((x_max - x_min) / resolution)
    H = int((y_max - y_min) / resolution)

    # Filter to range
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    )
    pts = pts[mask]

    # Pixel indices: x→cols (left=0), y flipped→rows
    col = ((pts[:, 0] - x_min) / resolution).astype(int)
    row = ((y_max - pts[:, 1]) / resolution).astype(int)

    # Clamp
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)

    density = np.zeros((H, W), dtype=np.float32)
    height  = np.full((H, W), -np.inf, dtype=np.float32)
    intensity = np.zeros((H, W), dtype=np.float32)

    np.add.at(density, (row, col), 1)
    np.maximum.at(height, (row, col), pts[:, 2])
    np.maximum.at(intensity, (row, col), pts[:, 3])

    # Normalise to [0, 255]
    def _norm(arr, vmin=None, vmax=None):
        arr = arr.copy()
        if vmin is None:
            vmin = arr[arr > -np.inf].min() if (arr > -np.inf).any() else 0
        if vmax is None:
            vmax = arr.max()
        arr = np.clip(arr, vmin, vmax)
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr[:] = 0
        return (arr * 255).astype(np.uint8)

    r = _norm(height, vmin=-2, vmax=2)        # height → red channel
    g = _norm(density, vmin=0, vmax=density.max() if density.max() else 1)
    b = _norm(intensity, vmin=0, vmax=1)

    # Where no points: leave black
    no_point = density == 0
    r[no_point] = 0
    g[no_point] = 0
    b[no_point] = 0

    return np.stack([r, g, b], axis=2)
