"""plot_grid_slam.py – Visualize RBPF / GMapping-style grid-SLAM output.

Static mode  : side-by-side figure — left: ground-truth map, right: best-particle
               SLAM map — with true path, estimated path and final particle cloud.
Animated mode: shows the SLAM map growing frame by frame next to the static GT map,
               with the particle cloud and robot arrow overlaid.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# ─────────────────────────────────────────────────────────────────────────────
# Data readers
# ─────────────────────────────────────────────────────────────────────────────

def read_trajectory(path: Path):
    steps, times, true_xy, est_xy, headings = [], [], [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            times.append(float(row["t"]))
            true_xy.append((float(row["x_true"]), float(row["y_true"])))
            est_xy.append((float(row["x_est"]),   float(row["y_est"])))
            headings.append(float(row["theta_est"]))
    return (np.array(steps), np.array(times),
            np.array(true_xy), np.array(est_xy), np.array(headings))


def read_particles(path: Path):
    """Returns {step: [(x, y, theta, weight), …]}."""
    by_step: dict = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            step = int(row["step"])
            by_step.setdefault(step, []).append((
                float(row["x"]), float(row["y"]),
                float(row["theta"]), float(row["weight"]),
            ))
    return by_step


def read_map(path: Path):
    """Returns arrays (world_x, world_y, prob) for all non-zero cells."""
    wx, wy, prob = [], [], []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            wx.append(float(row["world_x"]))
            wy.append(float(row["world_y"]))
            prob.append(float(row["prob"]))
    return np.array(wx), np.array(wy), np.array(prob)


def read_map_snapshots(path: Path):
    """Returns {step: (world_x, world_y, prob)} for incremental map frames."""
    by_step: dict = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            step = int(row["step"])
            by_step.setdefault(step, ([], [], []))
            by_step[step][0].append(float(row["world_x"]))
            by_step[step][1].append(float(row["world_y"]))
            by_step[step][2].append(float(row["prob"]))
    return {s: (np.array(x), np.array(y), np.array(p))
            for s, (x, y, p) in by_step.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_step(step, by_step):
    keys = sorted(k for k in by_step if k <= step)
    return keys[-1] if keys else None


def _draw_robot_arrow(ax, x, y, theta, color="blue"):
    ax.arrow(x, y, 0.5 * np.cos(theta), 0.5 * np.sin(theta),
             head_width=0.25, head_length=0.3, color=color, alpha=0.9, zorder=6)


def _scatter_map(ax, wx, wy, prob, resolution=0.2, title=""):
    """Render the occupancy map as a scatter plot on the given axes.
    Grey-scale: occupied → black (prob→1), free → white (prob→0), unknown (0.5) → grey.
    """
    if len(wx) == 0:
        return
    grey = 1.0 - np.clip(prob, 0.0, 1.0)
    rgba = np.column_stack([grey, grey, grey, np.ones_like(grey)])
    ax.scatter(wx, wy, c=rgba, s=(resolution * 72) ** 2,
               marker="s", linewidths=0, zorder=1)
    if title:
        ax.set_title(title, fontsize=10)


def _setup_ax(ax, label):
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(label, fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Static plot  – side-by-side: GT map (left) vs SLAM map (right)
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(steps, true_xy, est_xy, headings,
                particles_by_step,
                slam_wx, slam_wy, slam_prob,
                gt_wx,   gt_wy,   gt_prob,
                output: Path,
                resolution: float):

    fig, (ax_gt, ax_slam) = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle("Grid-SLAM (RBPF) – Final Map Comparison", fontsize=13)

    # ── Left panel: ground-truth map ─────────────────────────────────────────
    _scatter_map(ax_gt, gt_wx, gt_wy, gt_prob, resolution)
    ax_gt.plot(true_xy[:, 0], true_xy[:, 1], "g-", lw=1.5, label="True path", zorder=3)
    _draw_robot_arrow(ax_gt, true_xy[-1, 0], true_xy[-1, 1], headings[-1], color="green")
    _setup_ax(ax_gt, "Ground-Truth Map + True Path")
    ax_gt.legend(fontsize=8)

    # ── Right panel: SLAM map ────────────────────────────────────────────────
    _scatter_map(ax_slam, slam_wx, slam_wy, slam_prob, resolution)
    ax_slam.plot(true_xy[:, 0], true_xy[:, 1], "g--", lw=1.2,
                 alpha=0.5, label="True path", zorder=3)
    ax_slam.plot(est_xy[:, 0],  est_xy[:, 1],  "b",   lw=1.5,
                 label="RBPF estimate (best particle)", zorder=4)

    last_step = steps[-1]
    pkey = _nearest_step(last_step, particles_by_step)
    if pkey is not None:
        pts = np.array([(x, y) for x, y, _, _ in particles_by_step[pkey]])
        ax_slam.scatter(pts[:, 0], pts[:, 1], c="orange", s=12, alpha=0.6,
                        label="Particles (final)", zorder=5)

    _draw_robot_arrow(ax_slam, est_xy[-1, 0], est_xy[-1, 1], headings[-1])
    _setup_ax(ax_slam, "SLAM Map (Best Particle) + Estimated Path")
    ax_slam.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Animation  – GT map static on left; SLAM map rebuilt from periodic snapshots
#              on right, so the map visibly grows as the robot explores.
# ─────────────────────────────────────────────────────────────────────────────

def animate_results(steps, true_xy, est_xy, headings,
                    particles_by_step,
                    map_snapshots,          # {step: (wx, wy, prob)}
                    gt_wx,   gt_wy,   gt_prob,
                    resolution: float):

    fig, (ax_gt, ax_slam) = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle("Grid-SLAM (RBPF) – Animation", fontsize=13)

    # Compute shared axis limits from GT map + trajectories
    all_x = np.concatenate([true_xy[:, 0], est_xy[:, 0],
                             gt_wx if len(gt_wx) else np.array([0.0])])
    all_y = np.concatenate([true_xy[:, 1], est_xy[:, 1],
                             gt_wy if len(gt_wy) else np.array([0.0])])
    pad   = 1.5
    xlim  = (all_x.min() - pad, all_x.max() + pad)
    ylim  = (all_y.min() - pad, all_y.max() + pad)

    for ax in (ax_gt, ax_slam):
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Left panel: static GT map drawn once as permanent background
    _scatter_map(ax_gt, gt_wx, gt_wy, gt_prob, resolution)
    # Right panel starts empty; map is added per-frame from snapshots
    snap_keys = sorted(map_snapshots.keys())

    def draw_frame(frame_idx):
        step = steps[frame_idx]

        # ── Clear dynamic artists (keep index-0 scatter on ax_gt = GT map) ──
        # ax_gt: keep collection[0] (GT map), remove the rest
        while len(ax_gt.lines) > 0:
            ax_gt.lines[0].remove()
        while len(ax_gt.collections) > 1:
            ax_gt.collections[1].remove()
        while len(ax_gt.patches) > 0:
            ax_gt.patches[0].remove()

        # ax_slam: clear everything – we re-draw the snapshot each frame
        while len(ax_slam.lines) > 0:
            ax_slam.lines[0].remove()
        while len(ax_slam.collections) > 0:
            ax_slam.collections[0].remove()
        while len(ax_slam.patches) > 0:
            ax_slam.patches[0].remove()

        # ── Left: growing true path + robot arrow ────────────────────────────
        ax_gt.plot(true_xy[: frame_idx + 1, 0], true_xy[: frame_idx + 1, 1],
                   "g-", lw=1.5, label="True path", zorder=3)
        _draw_robot_arrow(ax_gt, true_xy[frame_idx, 0], true_xy[frame_idx, 1],
                          headings[frame_idx], color="green")
        ax_gt.set_title(f"Ground-Truth Map  (step {step})", fontsize=10)
        ax_gt.set_xlabel("X (m)"); ax_gt.set_ylabel("Y (m)")
        ax_gt.grid(True, alpha=0.25)
        ax_gt.legend(loc="upper right", fontsize=8)

        # ── Right: SLAM map from nearest snapshot ≤ current step ────────────
        snap_step = next((k for k in reversed(snap_keys) if k <= step), None)
        if snap_step is not None:
            sx, sy, sp = map_snapshots[snap_step]
            _scatter_map(ax_slam, sx, sy, sp, resolution)

        # Trajectories overlaid on map
        ax_slam.plot(true_xy[: frame_idx + 1, 0], true_xy[: frame_idx + 1, 1],
                     "g--", lw=1.2, alpha=0.5, label="True path", zorder=3)
        ax_slam.plot(est_xy[: frame_idx + 1, 0],  est_xy[: frame_idx + 1, 1],
                     "b",   lw=1.5, label="RBPF estimate", zorder=4)

        pkey = _nearest_step(step, particles_by_step)
        if pkey is not None:
            pts = np.array([(x, y) for x, y, _, _ in particles_by_step[pkey]])
            ax_slam.scatter(pts[:, 0], pts[:, 1], c="orange", s=12, alpha=0.5,
                            label="Particles", zorder=5)

        _draw_robot_arrow(ax_slam, est_xy[frame_idx, 0], est_xy[frame_idx, 1],
                          headings[frame_idx])
        ax_slam.set_xlim(*xlim); ax_slam.set_ylim(*ylim)
        ax_slam.set_aspect("equal")
        ax_slam.set_title(f"SLAM Map – Best Particle  (step {step})", fontsize=10)
        ax_slam.set_xlabel("X (m)"); ax_slam.set_ylabel("Y (m)")
        ax_slam.grid(True, alpha=0.25)
        ax_slam.legend(loc="upper right", fontsize=8)

    return FuncAnimation(fig, draw_frame, frames=len(steps),
                         interval=80, repeat=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Grid-SLAM (RBPF) output")
    parser.add_argument("--trajectory",  type=Path,
                        default=Path("output/grid_slam_trajectory.csv"))
    parser.add_argument("--particles",   type=Path,
                        default=Path("output/grid_slam_particles.csv"))
    parser.add_argument("--map",         type=Path,
                        default=Path("output/grid_slam_map.csv"))
    parser.add_argument("--snapshots",   type=Path,
                        default=Path("output/grid_slam_map_snapshots.csv"))
    parser.add_argument("--gt-map",      type=Path,
                        default=Path("output/grid_slam_gt_map.csv"))
    parser.add_argument("--output",      type=Path,
                        default=Path("output/grid_slam_plot.png"))
    parser.add_argument("--resolution",  type=float, default=0.2,
                        help="Grid cell size in metres (for rendering)")
    parser.add_argument("--animate",     action="store_true")
    parser.add_argument("--show",        action="store_true")
    args = parser.parse_args()

    steps, times, true_xy, est_xy, headings = read_trajectory(args.trajectory)
    particles_by_step                        = read_particles(args.particles)
    slam_wx, slam_wy, slam_prob              = read_map(args.map)
    gt_wx,   gt_wy,   gt_prob               = read_map(args.gt_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.animate:
        map_snapshots = read_map_snapshots(args.snapshots)
        anim = animate_results(steps, true_xy, est_xy, headings,
                               particles_by_step,
                               map_snapshots,
                               gt_wx,   gt_wy,   gt_prob,
                               args.resolution)
        if args.show:
            plt.show()
        else:
            out_gif = args.output.with_suffix(".gif")
            anim.save(str(out_gif), writer="pillow", fps=15)
            print(f"Saved animation to {out_gif}")
    else:
        plot_static(steps, true_xy, est_xy, headings,
                    particles_by_step,
                    slam_wx, slam_wy, slam_prob,
                    gt_wx,   gt_wy,   gt_prob,
                    args.output, args.resolution)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
