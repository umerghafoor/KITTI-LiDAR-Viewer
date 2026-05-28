import numpy as np
from pathlib import Path


def load_velo_scan(path: str | Path) -> np.ndarray:
    """Load a Velodyne binary scan. Returns (N, 4): x, y, z, intensity."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts


def velo_to_image(pts: np.ndarray, P: np.ndarray, velo_to_rect: np.ndarray,
                  img_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project velodyne points to image coordinates.

    Returns:
        uv: (M, 2) pixel coordinates of points that land inside the image
        depth: (M,) depths in camera frame for those points
    """
    h, w = img_shape
    pts_hom = np.hstack([pts[:, :3], np.ones((len(pts), 1))])  # (N,4)

    cam = (velo_to_rect @ pts_hom.T)  # (4,N)

    # Keep only points in front of camera
    mask = cam[2] > 0
    cam = cam[:, mask]

    uv_hom = P @ cam  # (3,N)
    uv = uv_hom[:2] / uv_hom[2]  # (2,N)

    # Keep points inside image bounds
    in_bounds = (
        (uv[0] >= 0) & (uv[0] < w) &
        (uv[1] >= 0) & (uv[1] < h)
    )
    return uv[:, in_bounds].T, cam[2, in_bounds]
