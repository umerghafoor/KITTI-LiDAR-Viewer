#!/usr/bin/env python3
"""
KITTI LiDAR Point Cloud Viewer
-------------------------------
Interactive viewer with three panels:
  1. Bird's-eye view (BEV) of the point cloud
  2. Front camera image with LiDAR depth overlay
  3. 3-D scatter plot (colour = height)

Controls
--------
  space            : play / pause
  n / right-arrow  : next frame
  p / left-arrow   : previous frame
  q / Escape       : quit
  s                : save current figure as PNG

Usage
-----
  python viewer.py [--drive DRIVE_DIR] [--calib CALIB_DIR] [--cam CAM] [--start N] [--fps N]

Defaults point to the KITTI sequence included in ~/Datasets/KITTI/extract/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # requires PyQt6 (or PyQt5); install with: pip install pyqt6
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from PIL import Image

# Ensure the kitti package in this repo is importable
sys.path.insert(0, str(Path(__file__).parent))
from kitti import build_projection_matrix, load_velo_scan, velo_to_image, make_bev

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("/home/umerghafoor/Datasets/KITTI/extract")
DEFAULT_DRIVE = (
    DATASET_ROOT
    / "2011_09_26_drive_0019_sync"
    / "2011_09_26"
    / "2011_09_26_drive_0019_sync"
)
DEFAULT_CALIB = DATASET_ROOT / "2011_09_26_calib" / "2011_09_26"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sorted_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(directory.glob(f"*{suffix}"))


def load_frame(frame_idx: int, velo_files: list, cam_files: list,
               P: np.ndarray, velo_to_rect: np.ndarray, cam: int):
    pts = load_velo_scan(velo_files[frame_idx])
    img = np.array(Image.open(cam_files[frame_idx]).convert("RGB"))
    bev = make_bev(pts)
    uv, depth = velo_to_image(pts, P, velo_to_rect, img.shape[:2])
    return pts, img, bev, uv, depth


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class KITTIViewer:
    def __init__(self, drive_dir: Path, calib_dir: Path, cam: int = 2,
                 start: int = 0, fps: int = 10):
        self.velo_files = sorted_files(drive_dir / "velodyne_points" / "data", ".bin")
        self.cam_files  = sorted_files(drive_dir / f"image_0{cam}" / "data", ".png")
        n = min(len(self.velo_files), len(self.cam_files))
        self.velo_files = self.velo_files[:n]
        self.cam_files  = self.cam_files[:n]
        self.n_frames   = n

        self.P, self.velo_to_rect = build_projection_matrix(calib_dir, cam)
        self.cam = cam
        self.fps = fps
        self.frame_idx = max(0, min(start, n - 1))
        self._playing = False
        self._timer = None

        self._build_ui()
        self._draw(self.frame_idx)

    # ------------------------------------------------------------------

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("KITTI LiDAR Viewer")

        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            hspace=0.35,
            wspace=0.05,
            top=0.93,
            bottom=0.12,
            left=0.03,
            right=0.97,
        )

        # Top-left: BEV
        self.ax_bev = self.fig.add_subplot(gs[0:2, 0])
        self.ax_bev.set_facecolor("#0d0d1a")
        self.ax_bev.set_title("Bird's-Eye View", color="white", fontsize=9)
        self.ax_bev.axis("off")

        # Top-middle: camera
        self.ax_cam = self.fig.add_subplot(gs[0:2, 1])
        self.ax_cam.set_facecolor("#0d0d1a")
        self.ax_cam.set_title(f"Camera {self.cam:02d} + LiDAR depth", color="white", fontsize=9)
        self.ax_cam.axis("off")

        # Top-right: 3-D scatter
        self.ax_3d = self.fig.add_subplot(gs[0:2, 2], projection="3d")
        self.ax_3d.set_facecolor("#0d0d1a")
        self.ax_3d.set_title("3-D Point Cloud", color="white", fontsize=9)
        self.ax_3d.tick_params(colors="white", labelsize=6)
        for pane in (self.ax_3d.xaxis.pane, self.ax_3d.yaxis.pane, self.ax_3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")

        # Bottom: info bar
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.set_facecolor("#1a1a2e")
        self.ax_info.axis("off")
        self.txt_info = self.ax_info.text(
            0.5, 0.5, "", transform=self.ax_info.transAxes,
            ha="center", va="center", color="lightgrey", fontsize=9,
            fontfamily="monospace",
        )

        # Keyboard + button callbacks
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Prev / Play-Pause / Next buttons
        ax_prev  = self.fig.add_axes([0.30, 0.03, 0.07, 0.05])
        ax_play  = self.fig.add_axes([0.38, 0.03, 0.09, 0.05])
        ax_next  = self.fig.add_axes([0.48, 0.03, 0.07, 0.05])
        self.btn_prev = Button(ax_prev,  "◀ Prev",   color="#2d2d5b", hovercolor="#4444aa")
        self.btn_play = Button(ax_play,  "▶  Play",  color="#1a3a1a", hovercolor="#2a6a2a")
        self.btn_next = Button(ax_next,  "Next ▶",   color="#2d2d5b", hovercolor="#4444aa")
        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white")
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())

        # Frame slider
        ax_slider = self.fig.add_axes([0.15, 0.05, 0.70, 0.025])
        self.slider = Slider(
            ax_slider, "Frame", 0, self.n_frames - 1,
            valinit=self.frame_idx, valstep=1,
            color="#4444aa",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # Save button
        ax_save = self.fig.add_axes([0.88, 0.03, 0.08, 0.05])
        self.btn_save = Button(ax_save, "Save PNG", color="#2d2d5b", hovercolor="#4444aa")
        self.btn_save.label.set_color("white")
        self.btn_save.on_clicked(self._save)

    # ------------------------------------------------------------------

    def _draw(self, frame_idx: int):
        pts, img, bev, uv, depth = load_frame(
            frame_idx, self.velo_files, self.cam_files,
            self.P, self.velo_to_rect, self.cam,
        )

        # --- BEV ---
        self.ax_bev.cla()
        self.ax_bev.set_facecolor("#0d0d1a")
        self.ax_bev.set_title("Bird's-Eye View  (R=height G=density B=intensity)",
                               color="white", fontsize=7)
        self.ax_bev.imshow(bev, origin="upper", aspect="auto")
        # Ego-vehicle dot
        h_px, w_px = bev.shape[:2]
        ego_col = int(0 / 0.1)          # x=0 → leftmost column
        ego_row = h_px // 2
        self.ax_bev.plot(ego_col, ego_row, "yo", markersize=5)
        self.ax_bev.set_xlabel("← Left          Right →", color="grey", fontsize=6)
        self.ax_bev.set_ylabel("Distance (m)", color="grey", fontsize=6)
        self.ax_bev.tick_params(colors="grey", labelsize=5)

        # --- Camera with depth overlay ---
        self.ax_cam.cla()
        self.ax_cam.set_facecolor("#0d0d1a")
        self.ax_cam.set_title(f"Camera {self.cam:02d} + LiDAR depth", color="white", fontsize=9)
        self.ax_cam.imshow(img)
        if len(uv):
            sc = self.ax_cam.scatter(
                uv[:, 0], uv[:, 1],
                c=depth, cmap="jet",
                vmin=depth.min(), vmax=min(depth.max(), 60),
                s=1, alpha=0.8,
            )
        self.ax_cam.axis("off")

        # --- 3-D scatter (subsample for speed) ---
        self.ax_3d.cla()
        self.ax_3d.set_facecolor("#0d0d1a")
        self.ax_3d.set_title("3-D Point Cloud", color="white", fontsize=9)
        step = max(1, len(pts) // 5000)
        sub = pts[::step]
        z = sub[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        self.ax_3d.scatter(
            sub[:, 0], sub[:, 1], sub[:, 2],
            c=z_norm, cmap="viridis",
            s=0.3, alpha=0.6,
        )
        self.ax_3d.set_xlabel("X (fwd)", color="white", fontsize=6, labelpad=1)
        self.ax_3d.set_ylabel("Y (left)", color="white", fontsize=6, labelpad=1)
        self.ax_3d.set_zlabel("Z (up)", color="white", fontsize=6, labelpad=1)
        self.ax_3d.tick_params(colors="white", labelsize=5)
        self.ax_3d.set_xlim(0, 60)
        self.ax_3d.set_ylim(-30, 30)
        self.ax_3d.set_zlim(-3, 5)

        # --- Info bar ---
        fname = self.velo_files[frame_idx].stem
        self.txt_info.set_text(
            f"Frame {frame_idx + 1}/{self.n_frames}   |   file: {fname}   |   "
            f"points: {len(pts):,}   |   projected: {len(uv):,}   |   "
            f"keys: [space] play/pause  [n] next  [p] prev  [s] save  [q] quit"
        )

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------

    def _step(self, delta: int):
        self.frame_idx = (self.frame_idx + delta) % self.n_frames
        self.slider.set_val(self.frame_idx)   # triggers _on_slider → _draw

    def _on_slider(self, val):
        idx = int(round(val))
        if idx != self.frame_idx:
            self.frame_idx = idx
            self._draw(self.frame_idx)

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            if self._timer is not None:
                self._timer.stop()
            self.btn_play.label.set_text("▶  Play")
            self.btn_play.ax.set_facecolor("#1a3a1a")
        else:
            self._playing = True
            self.btn_play.label.set_text("⏸ Pause")
            self.btn_play.ax.set_facecolor("#3a1a1a")
            interval_ms = max(1, int(1000 / self.fps))
            self._timer = self.fig.canvas.new_timer(interval=interval_ms)
            self._timer.add_callback(self._autoplay_tick)
            self._timer.start()
        self.fig.canvas.draw_idle()

    def _autoplay_tick(self):
        self.frame_idx = (self.frame_idx + 1) % self.n_frames
        # Update slider position without triggering the slider callback
        self.slider.eventson = False
        self.slider.set_val(self.frame_idx)
        self.slider.eventson = True
        self._draw(self.frame_idx)

    def _on_key(self, event):
        if event.key in ("n", "right"):
            self._step(+1)
        elif event.key in ("p", "left"):
            self._step(-1)
        elif event.key == " ":
            self._toggle_play()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "s":
            self._save(None)

    def _save(self, _event):
        out = Path(f"kitti_frame_{self.frame_idx:04d}.png")
        self.fig.savefig(out, dpi=150, facecolor=self.fig.get_facecolor())
        print(f"Saved → {out.resolve()}")

    # ------------------------------------------------------------------

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KITTI LiDAR Viewer")
    parser.add_argument("--drive", type=Path, default=DEFAULT_DRIVE)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--cam",   type=int,  default=2,
                        help="Camera index: 0=grey-left 1=grey-right 2=colour-left 3=colour-right")
    parser.add_argument("--start", type=int,  default=0,  help="Starting frame index")
    parser.add_argument("--fps",   type=int,  default=10, help="Autoplay speed (frames per second)")
    args = parser.parse_args()

    v = KITTIViewer(args.drive, args.calib, args.cam, args.start, args.fps)
    v.show()


if __name__ == "__main__":
    main()
