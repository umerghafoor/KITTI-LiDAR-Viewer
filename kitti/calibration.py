import numpy as np
from pathlib import Path


def _parse_value(s: str) -> np.ndarray | None:
    try:
        return np.array([float(x) for x in s.strip().split()])
    except ValueError:
        return None


def load_calib_cam_to_cam(path: str | Path) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            vals = _parse_value(val)
            if vals is None:
                continue
            if key.startswith("K_") or key.startswith("R_rect_"):
                data[key] = vals.reshape(3, 3)
            elif key.startswith("P_rect_"):
                data[key] = vals.reshape(3, 4)
            elif key.startswith("R_") and not key.startswith("R_rect"):
                data[key] = vals.reshape(3, 3)
            elif key.startswith("T_"):
                data[key] = vals.reshape(3, 1) if vals.size == 3 else vals
            elif key.startswith("S_"):
                data[key] = vals
            else:
                data[key] = vals
    return data


def load_calib_velo_to_cam(path: str | Path) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            vals = _parse_value(val)
            if key == "R":
                data["R"] = vals.reshape(3, 3)
            elif key == "T":
                data["T"] = vals.reshape(3, 1)
            else:
                data[key] = vals
    return data


def build_projection_matrix(calib_dir: str | Path, cam: int = 2):
    """Return (P, velo_to_cam_rect) where P is 3×4 and maps rectified cam coords to image."""
    calib_dir = Path(calib_dir)
    c2c = load_calib_cam_to_cam(calib_dir / "calib_cam_to_cam.txt")
    v2c = load_calib_velo_to_cam(calib_dir / "calib_velo_to_cam.txt")

    R_rect = c2c[f"R_rect_0{cam}"]   # 3×3
    P_rect = c2c[f"P_rect_0{cam}"]   # 3×4

    R = v2c["R"]   # 3×3
    T = v2c["T"]   # 3×1

    # Velodyne → cam0 rect
    Rt = np.hstack([R, T])            # 3×4
    R_rect_4x4 = np.eye(4)
    R_rect_4x4[:3, :3] = R_rect

    Rt_4x4 = np.eye(4)
    Rt_4x4[:3, :] = Rt

    velo_to_rect = R_rect_4x4 @ Rt_4x4  # 4×4
    return P_rect, velo_to_rect
