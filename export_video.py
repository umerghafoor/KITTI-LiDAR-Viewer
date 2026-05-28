#!/usr/bin/env python3
"""
KITTI LiDAR → MP4 exporter
----------------------------
Renders every frame (BEV + camera-depth + 3-D scatter) and encodes to MP4
using ffmpeg.  No display required — runs fully headless.

Usage
-----
  python export_video.py [--out OUTPUT.mp4] [--fps N] [--dpi N]
                         [--drive DIR] [--calib DIR] [--cam N]
                         [--start N] [--end N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from kitti import build_projection_matrix, load_velo_scan, velo_to_image, make_bev

DATASET_ROOT  = Path("/home/umerghafoor/Datasets/KITTI/extract")
DEFAULT_DRIVE = (
    DATASET_ROOT
    / "2011_09_26_drive_0019_sync"
    / "2011_09_26"
    / "2011_09_26_drive_0019_sync"
)
DEFAULT_CALIB = DATASET_ROOT / "2011_09_26_calib" / "2011_09_26"


def sorted_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(directory.glob(f"*{suffix}"))


def build_frame(frame_idx: int, velo_files, cam_files, P, velo_to_rect, cam,
                fig, ax_bev, ax_cam, ax_3d, ax_info):
    pts = load_velo_scan(velo_files[frame_idx])
    img = np.array(Image.open(cam_files[frame_idx]).convert("RGB"))
    bev = make_bev(pts)
    uv, depth = velo_to_image(pts, P, velo_to_rect, img.shape[:2])

    n_frames = len(velo_files)

    # BEV
    ax_bev.cla()
    ax_bev.set_facecolor("#0d0d1a")
    ax_bev.imshow(bev, origin="upper", aspect="auto")
    h_px = bev.shape[0]
    ax_bev.plot(0, h_px // 2, "yo", markersize=4)
    ax_bev.set_title("Bird's-Eye View  (R=height  G=density  B=intensity)",
                     color="white", fontsize=7)
    ax_bev.set_xlabel("← Left     Right →", color="grey", fontsize=6)
    ax_bev.tick_params(colors="grey", labelsize=5)

    # Camera + depth
    ax_cam.cla()
    ax_cam.set_facecolor("#0d0d1a")
    ax_cam.imshow(img)
    if len(uv):
        ax_cam.scatter(uv[:, 0], uv[:, 1],
                       c=depth, cmap="jet",
                       vmin=depth.min(), vmax=min(depth.max(), 60),
                       s=1, alpha=0.8)
    ax_cam.set_title(f"Camera {cam:02d} + LiDAR depth", color="white", fontsize=9)
    ax_cam.axis("off")

    # 3-D scatter
    ax_3d.cla()
    ax_3d.set_facecolor("#0d0d1a")
    step = max(1, len(pts) // 5000)
    sub  = pts[::step]
    z    = sub[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    ax_3d.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                  c=z_norm, cmap="viridis", s=0.3, alpha=0.6)
    ax_3d.set_xlabel("X (fwd)", color="white", fontsize=6, labelpad=1)
    ax_3d.set_ylabel("Y (left)", color="white", fontsize=6, labelpad=1)
    ax_3d.set_zlabel("Z (up)", color="white", fontsize=6, labelpad=1)
    ax_3d.tick_params(colors="white", labelsize=5)
    ax_3d.set_xlim(0, 60); ax_3d.set_ylim(-30, 30); ax_3d.set_zlim(-3, 5)
    ax_3d.set_title("3-D Point Cloud", color="white", fontsize=9)
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")

    # Info bar
    ax_info.cla()
    ax_info.set_facecolor("#1a1a2e")
    ax_info.axis("off")
    ax_info.text(0.5, 0.5,
                 f"Frame {frame_idx + 1}/{n_frames}   |   "
                 f"file: {velo_files[frame_idx].stem}   |   "
                 f"points: {len(pts):,}   |   projected: {len(uv):,}",
                 transform=ax_info.transAxes,
                 ha="center", va="center",
                 color="lightgrey", fontsize=9, fontfamily="monospace")


def main():
    parser = argparse.ArgumentParser(description="Export KITTI frames to MP4")
    parser.add_argument("--out",   default="kitti_lidar.mp4", help="Output file")
    parser.add_argument("--fps",   type=int,   default=10,    help="Frames per second")
    parser.add_argument("--dpi",   type=int,   default=120,   help="DPI (higher = larger file)")
    parser.add_argument("--drive", type=Path,  default=DEFAULT_DRIVE)
    parser.add_argument("--calib", type=Path,  default=DEFAULT_CALIB)
    parser.add_argument("--cam",   type=int,   default=2)
    parser.add_argument("--start", type=int,   default=0)
    parser.add_argument("--end",   type=int,   default=None,  help="Last frame (exclusive); default=all")
    args = parser.parse_args()

    velo_files = sorted_files(args.drive / "velodyne_points" / "data", ".bin")
    cam_files  = sorted_files(args.drive / f"image_0{args.cam}" / "data", ".png")
    n = min(len(velo_files), len(cam_files))
    velo_files = velo_files[args.start : args.end or n]
    cam_files  = cam_files [args.start : args.end or n]
    n_frames   = len(velo_files)

    P, velo_to_rect = build_projection_matrix(args.calib, args.cam)

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.35, wspace=0.05,
                            top=0.93, bottom=0.08, left=0.03, right=0.97)
    ax_bev  = fig.add_subplot(gs[0:2, 0])
    ax_cam  = fig.add_subplot(gs[0:2, 1])
    ax_3d   = fig.add_subplot(gs[0:2, 2], projection="3d")
    ax_info = fig.add_subplot(gs[2, :])
    ax_bev.set_facecolor("#0d0d1a")
    ax_cam.set_facecolor("#0d0d1a")
    ax_3d .set_facecolor("#0d0d1a")
    ax_info.set_facecolor("#1a1a2e")

    def animate(i):
        pct = (i + 1) / n_frames * 100
        print(f"\r  Rendering {i + 1}/{n_frames}  ({pct:.0f}%)", end="", flush=True)
        build_frame(i, velo_files, cam_files, P, velo_to_rect, args.cam,
                    fig, ax_bev, ax_cam, ax_3d, ax_info)

    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p"],  # broad player compatibility
    )

    print(f"Rendering {n_frames} frames → {args.out}  (fps={args.fps}, dpi={args.dpi})")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)
    anim.save(args.out, writer=writer, dpi=args.dpi,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    print(f"\nSaved → {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
