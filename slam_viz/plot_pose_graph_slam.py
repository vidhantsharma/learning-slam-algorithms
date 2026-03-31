"""Pose Graph SLAM visualisation – static plot + optimisation animation."""
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


def read_edges(path: Path):
    odom, lc = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pair = (int(row["from"]), int(row["to"]))
            if row["type"] == "loop_closure":
                lc.append(pair)
            else:
                odom.append(pair)
    return odom, lc


def read_iterations(path: Path):
    """Returns dict: iteration → (N, 2) array of (x, y)."""
    by_iter = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            it = int(row["iter"])
            node = int(row["node"])
            by_iter[it][node] = (float(row["x"]), float(row["y"]))
    result = {}
    for it, nodes in sorted(by_iter.items()):
        n = max(nodes.keys()) + 1
        arr = np.zeros((n, 2))
        for idx, (x, y) in nodes.items():
            arr[idx] = [x, y]
        result[it] = arr
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_edges(ax, xy, edges, **kwargs):
    for (i, j) in edges:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], **kwargs)


def _setup_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Static plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(true_xy, init_xy, opt_xy, odom_edges, lc_edges, output: Path,
                show: bool):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: before optimisation ─────────────────────────────────────────
    ax = axes[0]
    _setup_ax(ax, "Before Optimisation (odometry only)")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5, label="Ground truth")
    ax.plot(init_xy[:, 0], init_xy[:, 1], "r-", lw=1.2, label="Odometry (drifted)")
    _draw_edges(ax, init_xy, lc_edges, color="tab:purple", lw=0.6, ls="--",
                alpha=0.5)
    # Mark loop-closure nodes
    lc_nodes = set()
    for (i, j) in lc_edges:
        lc_nodes.add(i)
        lc_nodes.add(j)
    if lc_nodes:
        lc_idx = sorted(lc_nodes)
        ax.scatter(init_xy[lc_idx, 0], init_xy[lc_idx, 1], c="purple",
                   s=15, zorder=5, label="Loop-closure nodes")
    ax.scatter(init_xy[0, 0], init_xy[0, 1], c="green", s=80, marker="^",
               zorder=6, label="Start")
    ax.legend(fontsize=8, loc="best")

    # ── Right: after optimisation ─────────────────────────────────────────
    ax = axes[1]
    _setup_ax(ax, "After Gauss-Newton Optimisation")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5, label="Ground truth")
    ax.plot(opt_xy[:, 0], opt_xy[:, 1], "b-", lw=1.2, label="Optimised")
    _draw_edges(ax, opt_xy, lc_edges, color="tab:purple", lw=0.6, ls="--",
                alpha=0.5)
    if lc_nodes:
        ax.scatter(opt_xy[lc_idx, 0], opt_xy[lc_idx, 1], c="purple",
                   s=15, zorder=5, label="Loop-closure nodes")
    ax.scatter(opt_xy[0, 0], opt_xy[0, 1], c="green", s=80, marker="^",
               zorder=6, label="Start")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Pose Graph SLAM — Loop-Closure Correction", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Live: interactive side-by-side real-time comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_live(true_xy, init_xy, opt_xy, lc_edges, iterations,
              frame_delay: float = 0.9):
    """Two-panel interactive display.  Left = frozen odometry.
    Right = Gauss-Newton iterating live.  No file is saved."""
    it_keys = sorted(iterations.keys())

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pose Graph SLAM — Gauss-Newton (real-time)",
                 fontsize=14, fontweight="bold")

    # ── Left panel: frozen odometry ───────────────────────────────────────
    ax_l = axes[0]
    _setup_ax(ax_l, "Odometry only  (dead-reckoning)")
    ax_l.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5,
              label="Ground truth")
    ax_l.plot(init_xy[:, 0], init_xy[:, 1], "r-", lw=1.2,
              label="Odometry (drifted)")
    _draw_edges(ax_l, init_xy, lc_edges,
                color="tab:purple", lw=0.6, ls="--", alpha=0.4)
    lc_nodes = sorted({n for e in lc_edges for n in e})
    if lc_nodes:
        ax_l.scatter(init_xy[lc_nodes, 0], init_xy[lc_nodes, 1],
                     c="purple", s=15, zorder=5, label="LC nodes")
    ax_l.scatter(init_xy[0, 0], init_xy[0, 1], c="green", s=90,
                 marker="^", zorder=6, label="Start")
    ax_l.legend(fontsize=8, loc="best")

    # ── Right panel: live optimisation ────────────────────────────────────
    ax_r = axes[1]
    _setup_ax(ax_r, "Gauss-Newton — waiting…")
    ax_r.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5,
              label="Ground truth")
    (line_opt,) = ax_r.plot([], [], "b-", lw=1.5, label="Current estimate",
                             zorder=3)
    scat_nodes = ax_r.scatter([], [], c="tab:blue", s=14, zorder=4)
    lc_lines_r = []
    for _ in lc_edges:
        (ln,) = ax_r.plot([], [], color="tab:purple", lw=0.6, ls="--",
                          alpha=0.4)
        lc_lines_r.append(ln)
    ax_r.scatter(true_xy[0, 0], true_xy[0, 1], c="green", s=90,
                 marker="^", zorder=6, label="Start")
    ax_r.legend(fontsize=8, loc="best")

    # Fixed axis limits (use odometry extent so both panels match)
    all_xy = np.vstack([true_xy, init_xy])
    margin = 2.0
    for ax in axes:
        ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
        ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.5)

    # ── Animate iterations ────────────────────────────────────────────────
    for it in it_keys:
        xy = iterations[it]
        line_opt.set_data(xy[:, 0], xy[:, 1])
        scat_nodes.set_offsets(xy)
        for k, (i, j) in enumerate(lc_edges):
            lc_lines_r[k].set_data([xy[i, 0], xy[j, 0]],
                                    [xy[i, 1], xy[j, 1]])
        if it == 0:
            label = "Iter 0 — initial (odometry)"
        else:
            label = f"Iteration {it}"
        ax_r.set_title(f"Gauss-Newton — {label}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(frame_delay)

    ax_r.set_title("Converged  ✓")
    fig.canvas.draw()
    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Animation: show Gauss-Newton iterations converging
# ─────────────────────────────────────────────────────────────────────────────

def plot_animate(true_xy, init_xy, opt_xy, odom_edges, lc_edges, iterations,
                 output: Path, show: bool):
    it_keys = sorted(iterations.keys())

    fig, ax = plt.subplots(figsize=(10, 8))
    _setup_ax(ax, "")

    # Fixed elements
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5, label="Ground truth",
            zorder=1)

    (line_opt,) = ax.plot([], [], "b-", lw=1.5, label="Current estimate",
                          zorder=3)
    scat_nodes = ax.scatter([], [], c="tab:blue", s=12, zorder=4)
    lc_lines = []
    for _ in lc_edges:
        (ln,) = ax.plot([], [], color="tab:purple", lw=0.6, ls="--", alpha=0.5)
        lc_lines.append(ln)

    ax.scatter(true_xy[0, 0], true_xy[0, 1], c="green", s=80, marker="^",
               zorder=6, label="Start")
    ax.legend(fontsize=8, loc="best")

    # Set axis limits from the init (widest) trajectory
    all_xy = np.vstack([true_xy, init_xy])
    margin = 2.0
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    title_text = ax.set_title("")

    def update(frame):
        it = it_keys[frame]
        xy = iterations[it]
        line_opt.set_data(xy[:, 0], xy[:, 1])
        scat_nodes.set_offsets(xy)
        for k, (i, j) in enumerate(lc_edges):
            lc_lines[k].set_data([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]])
        label = "Initial (odometry)" if it == 0 else f"Iteration {it}"
        title_text.set_text(f"Pose Graph SLAM — Gauss-Newton — {label}")
        return [line_opt, scat_nodes, title_text] + lc_lines

    anim = FuncAnimation(fig, update, frames=len(it_keys),
                         interval=400, blit=False, repeat_delay=2000)
    anim.save(str(output), writer="pillow", dpi=120)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pose Graph SLAM visualisation")
    parser.add_argument("--animate", action="store_true",
                        help="Create GIF animation of GN iterations")
    parser.add_argument("--live", action="store_true",
                        help="Interactive real-time side-by-side comparison "
                             "(odometry vs optimising) in a live window")
    parser.add_argument("--speed", type=float, default=0.9,
                        help="Seconds between frames in --live mode (default 0.9)")
    parser.add_argument("--show", action="store_true",
                        help="Display plot window")
    args = parser.parse_args()

    traj_path = OUTPUT_DIR / "pose_graph_trajectory.csv"
    edge_path = OUTPUT_DIR / "pose_graph_edges.csv"
    iter_path = OUTPUT_DIR / "pose_graph_iterations.csv"

    true_xy, init_xy, opt_xy = read_trajectory(traj_path)
    odom_edges, lc_edges = read_edges(edge_path)

    if args.live:
        iterations = read_iterations(iter_path)
        plot_live(true_xy, init_xy, opt_xy, lc_edges, iterations,
                  frame_delay=args.speed)
        return  # live mode is self-contained

    plot_static(true_xy, init_xy, opt_xy, odom_edges, lc_edges,
                OUTPUT_DIR / "pose_graph_static.png", show=args.show)

    if args.animate:
        iterations = read_iterations(iter_path)
        plot_animate(true_xy, init_xy, opt_xy, odom_edges, lc_edges,
                     iterations,
                     OUTPUT_DIR / "pose_graph_animation.gif",
                     show=args.show)


if __name__ == "__main__":
    main()
