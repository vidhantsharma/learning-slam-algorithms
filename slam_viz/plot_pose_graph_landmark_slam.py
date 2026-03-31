"""Pose Graph SLAM with Landmarks – static plot + animation."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

OUTPUT_DIR = Path("output")

# ─────────────────────────────────────────────────────────────────────────────
# Data readers
# ─────────────────────────────────────────────────────────────────────────────

def read_trajectory(path: Path):
    true_xy, init_xy, opt_xy = [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            true_xy.append((float(row["x_true"]), float(row["y_true"])))
            init_xy.append((float(row["x_init"]), float(row["y_init"])))
            opt_xy.append((float(row["x_opt"]),  float(row["y_opt"])))
    return np.array(true_xy), np.array(init_xy), np.array(opt_xy)


def read_landmarks(path: Path):
    true_xy, init_xy, opt_xy = [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            true_xy.append((float(row["x_true"]), float(row["y_true"])))
            init_xy.append((float(row["x_init"]), float(row["y_init"])))
            opt_xy.append((float(row["x_opt"]),  float(row["y_opt"])))
    return np.array(true_xy), np.array(init_xy), np.array(opt_xy)


def read_edges(path: Path):
    odom, lm = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pair = (int(row["from"]), int(row["to"]))
            if row["type"] == "odometry":
                odom.append(pair)
            else:
                lm.append(pair)
    return odom, lm


def read_iterations(path: Path):
    """Returns (poses_by_iter, lms_by_iter) where each is dict int→(N,2) array."""
    pose_data = defaultdict(dict)
    lm_data   = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            it  = int(row["iter"])
            idx = int(row["idx"])
            xy  = (float(row["x"]), float(row["y"]))
            if row["type"] == "pose":
                pose_data[it][idx] = xy
            else:
                lm_data[it][idx] = xy

    def to_array(by_iter):
        result = {}
        for it, nodes in sorted(by_iter.items()):
            if not nodes:
                continue
            n = max(nodes.keys()) + 1
            arr = np.zeros((n, 2))
            for idx, (x, y) in nodes.items():
                arr[idx] = [x, y]
            result[it] = arr
        return result

    return to_array(pose_data), to_array(lm_data)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)


def _axis_limits(arrays, margin=1.5):
    all_pts = np.vstack(arrays)
    return (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin,
            all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)


# ─────────────────────────────────────────────────────────────────────────────
# Static plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(true_traj, init_traj, opt_traj,
                true_lms, init_lms, opt_lms,
                output: Path, show: bool):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xl, xr, yl, yr = _axis_limits(
        [true_traj, init_traj, opt_traj, true_lms, init_lms, opt_lms])

    for ax in axes:
        ax.set_xlim(xl, xr)
        ax.set_ylim(yl, yr)

    # ── Left: before optimisation ─────────────────────────────────────────
    ax = axes[0]
    _setup_ax(ax, "Before Optimisation (odometry + first-observation LM init)")
    ax.plot(true_traj[:, 0], true_traj[:, 1], "k--", lw=1.5,
            label="True trajectory")
    ax.plot(init_traj[:, 0], init_traj[:, 1], "r-", lw=1.2,
            label="Odometry (drifted)")
    ax.scatter(true_lms[:, 0], true_lms[:, 1],
               marker="*", c="black", s=120, zorder=5, label="True landmarks")
    ax.scatter(init_lms[:, 0], init_lms[:, 1],
               marker="x", c="tab:red", s=60, linewidths=1.5, zorder=6,
               label="Init landmark estimates")
    ax.scatter(init_traj[0, 0], init_traj[0, 1],
               c="green", s=100, marker="^", zorder=7, label="Start")
    ax.legend(fontsize=8, loc="best")

    # ── Right: after optimisation ─────────────────────────────────────────
    ax = axes[1]
    _setup_ax(ax, "After Gauss-Newton Optimisation (with LM damping)")
    ax.plot(true_traj[:, 0], true_traj[:, 1], "k--", lw=1.5,
            label="True trajectory")
    ax.plot(opt_traj[:, 0], opt_traj[:, 1], "b-", lw=1.2,
            label="Optimised trajectory")
    ax.scatter(true_lms[:, 0], true_lms[:, 1],
               marker="*", c="black", s=120, zorder=5, label="True landmarks")
    ax.scatter(opt_lms[:, 0], opt_lms[:, 1],
               marker="^", c="tab:blue", s=60, zorder=6,
               label="Optimised landmark estimates")
    # Draw error lines between true and optimised landmarks
    for i in range(len(true_lms)):
        ax.plot([true_lms[i, 0], opt_lms[i, 0]],
                [true_lms[i, 1], opt_lms[i, 1]],
                "tab:orange", lw=0.8, alpha=0.6)
    ax.scatter(opt_traj[0, 0], opt_traj[0, 1],
               c="green", s=100, marker="^", zorder=7, label="Start")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Pose Graph SLAM with Landmarks — Gauss-Newton + LM Damping",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Live interactive view: shows optimisation converging in real-time
# ─────────────────────────────────────────────────────────────────────────────

def plot_live(true_traj, init_traj, true_lms, init_lms,
              poses_iters, lms_iters, frame_delay: float = 0.6):
    it_keys = sorted(poses_iters.keys())

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pose Graph SLAM with Landmarks — Gauss-Newton (real-time)",
                 fontsize=13, fontweight="bold")

    xl, xr, yl, yr = _axis_limits(
        [true_traj, init_traj, true_lms, init_lms], margin=1.5)

    # ── Left panel: frozen dead-reckoned state ────────────────────────────
    ax_l = axes[0]
    _setup_ax(ax_l, "Odometry only  (dead-reckoning)")
    ax_l.plot(true_traj[:, 0], true_traj[:, 1], "k--", lw=1.5,
              label="True trajectory")
    ax_l.plot(init_traj[:, 0], init_traj[:, 1], "r-", lw=1.2,
              label="Odometry (drifted)")
    ax_l.scatter(true_lms[:, 0], true_lms[:, 1],
                 marker="*", c="black", s=120, zorder=5, label="True LMs")
    ax_l.scatter(init_lms[:, 0], init_lms[:, 1],
                 marker="x", c="tab:red", s=60, linewidths=1.5, zorder=6,
                 label="Init LM estimates")
    ax_l.scatter(init_traj[0, 0], init_traj[0, 1],
                 c="green", s=90, marker="^", zorder=7, label="Start")
    ax_l.set_xlim(xl, xr); ax_l.set_ylim(yl, yr)
    ax_l.legend(fontsize=8, loc="best")

    # ── Right panel: live optimisation ────────────────────────────────────
    ax_r = axes[1]
    _setup_ax(ax_r, "Gauss-Newton — waiting…")
    ax_r.plot(true_traj[:, 0], true_traj[:, 1], "k--", lw=1.5,
              label="True trajectory")
    (line_opt,) = ax_r.plot([], [], "b-", lw=1.5, label="Current trajectory",
                             zorder=3)
    scat_poses = ax_r.scatter([], [], c="tab:blue", s=12, zorder=4)
    scat_lms   = ax_r.scatter([], [], marker="^", c="tab:orange", s=60,
                               zorder=5, label="LM estimates")
    ax_r.scatter(true_lms[:, 0], true_lms[:, 1],
                 marker="*", c="black", s=120, zorder=6, label="True LMs")
    ax_r.scatter(true_traj[0, 0], true_traj[0, 1],
                 c="green", s=90, marker="^", zorder=7, label="Start")
    ax_r.set_xlim(xl, xr); ax_r.set_ylim(yl, yr)
    ax_r.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.5)

    for it in it_keys:
        pose_xy = poses_iters[it]
        lm_xy   = lms_iters.get(it, np.empty((0, 2)))
        line_opt.set_data(pose_xy[:, 0], pose_xy[:, 1])
        scat_poses.set_offsets(pose_xy)
        if lm_xy.shape[0] > 0:
            scat_lms.set_offsets(lm_xy)
        label = "Iter 0 — initial" if it == 0 else f"Iteration {it}"
        ax_r.set_title(f"Gauss-Newton — {label}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(frame_delay)

    ax_r.set_title("Converged  ✓")
    fig.canvas.draw()
    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Animation: GIF of Gauss-Newton iterations
# ─────────────────────────────────────────────────────────────────────────────

def plot_animate(true_traj, init_traj, true_lms,
                 poses_iters, lms_iters,
                 output: Path, show: bool):
    it_keys = sorted(poses_iters.keys())

    fig, ax = plt.subplots(figsize=(9, 7))
    _setup_ax(ax, "")

    xl, xr, yl, yr = _axis_limits([true_traj, init_traj, true_lms], margin=1.5)
    ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)

    ax.plot(true_traj[:, 0], true_traj[:, 1], "k--", lw=1.5,
            label="True trajectory", zorder=1)
    ax.scatter(true_lms[:, 0], true_lms[:, 1],
               marker="*", c="black", s=120, zorder=6, label="True landmarks")

    (line_opt,) = ax.plot([], [], "b-", lw=1.5, label="Optimised trajectory",
                          zorder=3)
    scat_poses = ax.scatter([], [], c="tab:blue", s=12, zorder=4)
    scat_lms   = ax.scatter([], [], marker="^", c="tab:orange", s=70,
                             zorder=5, label="LM estimates")

    ax.scatter(true_traj[0, 0], true_traj[0, 1],
               c="green", s=80, marker="^", zorder=7, label="Start")
    ax.legend(fontsize=8, loc="best")

    title_text = ax.set_title("")

    def update(frame):
        it = it_keys[frame]
        pose_xy = poses_iters[it]
        lm_xy   = lms_iters.get(it, np.empty((0, 2)))
        line_opt.set_data(pose_xy[:, 0], pose_xy[:, 1])
        scat_poses.set_offsets(pose_xy)
        if lm_xy.shape[0] > 0:
            scat_lms.set_offsets(lm_xy)
        label = "Initial (odometry)" if it == 0 else f"Iteration {it}"
        title_text.set_text(
            f"Pose Graph SLAM with Landmarks — Gauss-Newton — {label}")
        return [line_opt, scat_poses, scat_lms, title_text]

    anim = FuncAnimation(fig, update, frames=len(it_keys),
                         interval=350, blit=False, repeat_delay=2000)
    anim.save(str(output), writer="pillow", dpi=120)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pose Graph SLAM with Landmarks visualisation")
    parser.add_argument("--animate", action="store_true",
                        help="Create GIF animation of GN iterations")
    parser.add_argument("--live", action="store_true",
                        help="Interactive real-time comparison in a live window")
    parser.add_argument("--speed", type=float, default=0.6,
                        help="Seconds between frames in --live mode (default 0.6)")
    parser.add_argument("--show", action="store_true",
                        help="Display plot window")
    args = parser.parse_args()

    traj_path = OUTPUT_DIR / "pg_lm_trajectory.csv"
    lm_path   = OUTPUT_DIR / "pg_lm_landmarks.csv"
    edge_path = OUTPUT_DIR / "pg_lm_edges.csv"
    iter_path = OUTPUT_DIR / "pg_lm_iterations.csv"

    true_traj, init_traj, opt_traj = read_trajectory(traj_path)
    true_lms,  init_lms,  opt_lms  = read_landmarks(lm_path)
    _, _ = read_edges(edge_path)

    if args.live:
        poses_iters, lms_iters = read_iterations(iter_path)
        plot_live(true_traj, init_traj, true_lms, init_lms,
                  poses_iters, lms_iters, frame_delay=args.speed)
        return

    plot_static(true_traj, init_traj, opt_traj,
                true_lms, init_lms, opt_lms,
                OUTPUT_DIR / "pg_landmark_static.png",
                show=args.show)

    if args.animate:
        poses_iters, lms_iters = read_iterations(iter_path)
        plot_animate(true_traj, init_traj, true_lms,
                     poses_iters, lms_iters,
                     OUTPUT_DIR / "pg_landmark_animation.gif",
                     show=args.show)


if __name__ == "__main__":
    main()
