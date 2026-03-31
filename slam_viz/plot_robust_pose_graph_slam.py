#!/usr/bin/env python3
"""Visualization for Robust Pose Graph SLAM (Max-Mixture + Huber kernel).

Usage
-----
  python3 slam_viz/plot_robust_pose_graph_slam.py [--animate] [--show]

Modes
-----
  (default)  Static two-panel figure: before / after optimisation.
  --animate  Animate the Gauss-Newton iterations.
  --live     Alias for --animate.
  --show     Display the window instead of saving a PNG.

Edge colour coding (both panels)
---------------------------------
  grey        odometry edges
  steelblue   loop-closure — inlier (MM component 0 selected)
  red / dash  loop-closure — outlier suppressed (MM component 1 selected)
  orange      loop-closure — injected false outlier (for reference in the
              "before" panel only, to show the algorithm rejected it)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import pandas as pd

OUT_DIR = "output"
TRAJ_CSV   = os.path.join(OUT_DIR, "robust_trajectory.csv")
EDGES_CSV  = os.path.join(OUT_DIR, "robust_edges.csv")
ITERS_CSV  = os.path.join(OUT_DIR, "robust_iterations.csv")
STATIC_PNG = os.path.join(OUT_DIR, "robust_slam_static.png")
ANIM_GIF   = os.path.join(OUT_DIR, "robust_slam_animation.gif")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    traj  = pd.read_csv(TRAJ_CSV)
    edges = pd.read_csv(EDGES_CSV)
    iters = pd.read_csv(ITERS_CSV)
    return traj, edges, iters


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def edge_colour(row):
    """Return (colour, linestyle, zorder, alpha) for one edge row."""
    if row["type"] == "odometry":
        return ("grey", "-", 1, 0.4)
    # loop-closure
    if row["mm_selected"] == 0:
        if row["injected_outlier"] == 1:
            # The algorithm correctly suppressed it — show as orange dashed
            return ("orangered", "--", 2, 0.6)
        return ("steelblue", "-", 3, 0.8)
    else:
        # outlier component won → suppressed
        return ("crimson", "--", 2, 0.5)


def draw_trajectory(ax, xs, ys, label, colour, lw=1.5, alpha=0.9, zorder=5):
    ax.plot(xs, ys, colour, linewidth=lw, alpha=alpha, zorder=zorder, label=label)
    ax.plot(xs[0], ys[0], "o", color=colour, markersize=7, zorder=zorder + 1)
    ax.plot(xs[-1], ys[-1], "s", color=colour, markersize=7, zorder=zorder + 1)


def draw_edges(ax, nodes_x, nodes_y, edges):
    for _, row in edges.iterrows():
        i, j = int(row["from"]), int(row["to"])
        c, ls, zo, alpha = edge_colour(row)
        ax.plot([nodes_x[i], nodes_x[j]], [nodes_y[i], nodes_y[j]],
                color=c, linestyle=ls, linewidth=1.0, alpha=alpha, zorder=zo)


def legend_patches():
    return [
        mpatches.Patch(color="grey",       label="Odometry"),
        mpatches.Patch(color="steelblue",  label="LC — inlier (component 0 wins)"),
        mpatches.Patch(color="crimson",    label="LC — suppressed (null wins)"),
        mpatches.Patch(color="orangered",  label="LC — injected outlier (correctly suppressed)"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Static plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_static(show: bool):
    traj, edges, _ = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Robust Pose Graph SLAM (Max-Mixture + Huber Kernel)", fontsize=14)

    panels = [
        ("Before Optimisation",  traj["x_init"].values,  traj["y_init"].values,  "#d62728"),
        ("After Optimisation",   traj["x_opt"].values,   traj["y_opt"].values,   "#2ca02c"),
    ]

    node_arrays = [
        (traj["x_init"].values, traj["y_init"].values),
        (traj["x_opt"].values,  traj["y_opt"].values),
    ]

    for ax, (title, xs, ys, colour), (nx, ny) in zip(axes, panels, node_arrays):
        draw_edges(ax, nx, ny, edges)
        draw_trajectory(ax, traj["x_true"].values, traj["y_true"].values,
                        "Ground truth", "#1f77b4", lw=2)
        draw_trajectory(ax, xs, ys, "Robot trajectory", colour, lw=1.5)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    fig.legend(handles=legend_patches(),
               loc="lower center", ncol=4, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if show:
        plt.show()
    else:
        plt.savefig(STATIC_PNG, dpi=150, bbox_inches="tight")
        print(f"Saved {STATIC_PNG}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Animated plot — replay GN iterations
# ──────────────────────────────────────────────────────────────────────────────

def plot_animate(show: bool):
    traj, edges, iters = load_data()

    n_iters = iters["iter"].max() + 1
    n_nodes  = iters["node"].max() + 1

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Robust Pose Graph SLAM — Gauss-Newton Iterations")

    truth_x = traj["x_true"].values
    truth_y = traj["y_true"].values
    ax.plot(truth_x, truth_y, "#1f77b4", lw=2, label="Ground truth")

    traj_line, = ax.plot([], [], "#d62728", lw=1.5, label="Optimising…")
    iter_text  = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         fontsize=10, verticalalignment="top")
    ax.legend(handles=legend_patches() + [
        mpatches.Patch(color="#1f77b4", label="Ground truth"),
        mpatches.Patch(color="#d62728", label="Optimising…"),
    ], loc="upper right", fontsize=7)

    # Pre-compute per-iteration node arrays
    iter_data = []
    for it in range(n_iters):
        df = iters[iters["iter"] == it].sort_values("node")
        iter_data.append((df["x"].values, df["y"].values))

    # Edge line objects — drawn once, updated per frame
    edge_lines = []
    for _, row in edges.iterrows():
        c, ls, zo, alpha = edge_colour(row)
        ln, = ax.plot([], [], color=c, linestyle=ls,
                      linewidth=1.0, alpha=alpha, zorder=zo)
        edge_lines.append((ln, int(row["from"]), int(row["to"])))

    all_x = np.concatenate([traj["x_true"].values, traj["x_init"].values,
                             traj["x_opt"].values])
    all_y = np.concatenate([traj["y_true"].values, traj["y_init"].values,
                             traj["y_opt"].values])
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)

    def init_fn():
        traj_line.set_data([], [])
        iter_text.set_text("")
        for ln, _, _ in edge_lines:
            ln.set_data([], [])
        return [traj_line, iter_text] + [ln for ln, *_ in edge_lines]

    def update(frame):
        xs, ys = iter_data[frame]
        traj_line.set_data(xs, ys)
        iter_text.set_text(f"Iteration {frame + 1} / {n_iters}")
        for ln, fi, ti in edge_lines:
            ln.set_data([xs[fi], xs[ti]], [ys[fi], ys[ti]])
        return [traj_line, iter_text] + [ln for ln, *_ in edge_lines]

    ani = animation.FuncAnimation(
        fig, update, frames=n_iters, init_func=init_fn,
        interval=400, blit=True, repeat=True)

    if show:
        plt.show()
    else:
        ani.save(ANIM_GIF, writer="pillow", fps=3)
        print(f"Saved {ANIM_GIF}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise Robust Pose Graph SLAM")
    parser.add_argument("--animate", "--live", action="store_true",
                        help="Animate Gauss-Newton iterations")
    parser.add_argument("--show", action="store_true",
                        help="Show window instead of saving image")
    args = parser.parse_args()

    for path in (TRAJ_CSV, EDGES_CSV, ITERS_CSV):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run the C++ demo first.", file=sys.stderr)
            sys.exit(1)

    if args.animate:
        plot_animate(args.show)
    else:
        plot_static(args.show)


if __name__ == "__main__":
    main()
