"""Hierarchical Pose Graph SLAM visualisation – static plot + keyframe animation."""
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
    true_xy, init_xy, opt_xy, kf_mask = [], [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            true_xy.append((float(row["x_true"]),  float(row["y_true"])))
            init_xy.append((float(row["x_init"]),  float(row["y_init"])))
            opt_xy.append( (float(row["x_opt"]),   float(row["y_opt"])))
            kf_mask.append(int(row["is_keyframe"]) == 1)
    return (np.array(true_xy), np.array(init_xy),
            np.array(opt_xy),  np.array(kf_mask))


def read_kf_edges(path: Path):
    odom, lc = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pair = (int(row["from_kf"]), int(row["to_kf"]))
            if row["type"] == "loop_closure":
                lc.append(pair)
            else:
                odom.append(pair)
    return odom, lc


def read_iterations(path: Path):
    """Returns dict: iteration → (num_kf, 2) array of (x, y)."""
    by_iter = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            it   = int(row["iter"])
            kf   = int(row["kf"])
            by_iter[it][kf] = (float(row["x"]), float(row["y"]))
    result = {}
    for it, nodes in sorted(by_iter.items()):
        n = max(nodes.keys()) + 1
        arr = np.zeros((n, 2))
        for idx, (x, y) in nodes.items():
            arr[idx] = [x, y]
        result[it] = arr
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_kf_edges(ax, kf_xy, edges, **kwargs):
    for (i, j) in edges:
        ax.plot([kf_xy[i, 0], kf_xy[j, 0]],
                [kf_xy[i, 1], kf_xy[j, 1]], **kwargs)


def _setup_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Static plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(true_xy, init_xy, opt_xy, kf_mask,
                odom_kf_edges, lc_kf_edges, output: Path, show: bool):

    kf_init_xy = init_xy[kf_mask]
    kf_opt_xy  = opt_xy[kf_mask]
    lc_nodes   = sorted({n for pair in lc_kf_edges for n in pair})

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # ── Left: before optimisation ─────────────────────────────────────────
    ax = axes[0]
    _setup_ax(ax, "Before Optimisation  (dead-reckoned)")
    ax.plot(true_xy[:, 0],  true_xy[:, 1],  "k--", lw=1.5,  label="Ground truth")
    ax.plot(init_xy[:, 0],  init_xy[:, 1],  "r-",  lw=0.8,  alpha=0.6, label="Fine (drifted)")
    ax.scatter(kf_init_xy[:, 0], kf_init_xy[:, 1],
               c="tab:orange", s=25, zorder=5, label="Keyframes")
    _draw_kf_edges(ax, kf_init_xy, lc_kf_edges,
                   color="tab:purple", lw=0.7, ls="--", alpha=0.5)
    if lc_nodes:
        ax.scatter(kf_init_xy[lc_nodes, 0], kf_init_xy[lc_nodes, 1],
                   c="purple", s=40, zorder=6, label="LC keyframes")
    ax.scatter(*init_xy[0], c="green", s=80, marker="^", zorder=7, label="Start")
    ax.legend(fontsize=8, loc="best")

    # ── Right: after optimisation ─────────────────────────────────────────
    ax = axes[1]
    _setup_ax(ax, "After Hierarchical Gauss-Newton")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.5,  label="Ground truth")
    ax.plot(opt_xy[:, 0],  opt_xy[:, 1],  "b-",  lw=0.8,  alpha=0.7, label="Fine (corrected)")
    ax.scatter(kf_opt_xy[:, 0], kf_opt_xy[:, 1],
               c="tab:orange", s=25, zorder=5, label="Keyframes")
    _draw_kf_edges(ax, kf_opt_xy, lc_kf_edges,
                   color="tab:purple", lw=0.7, ls="--", alpha=0.5)
    if lc_nodes:
        ax.scatter(kf_opt_xy[lc_nodes, 0], kf_opt_xy[lc_nodes, 1],
                   c="purple", s=40, zorder=6, label="LC keyframes")
    ax.scatter(*opt_xy[0], c="green", s=80, marker="^", zorder=7, label="Start")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Hierarchical Pose Graph SLAM\n"
                 "Global graph: keyframes only  |  Propagation: fine nodes re-rolled from corrected keyframes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output), dpi=150)
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Animation – keyframe graph evolving over GN iterations
# ─────────────────────────────────────────────────────────────────────────────

def animate(true_xy, opt_xy, kf_mask,
            odom_kf_edges, lc_kf_edges, iters: dict, output: Path, show: bool):
    if not iters:
        print("No iteration data – skipping animation.")
        return

    iter_list = sorted(iters.keys())
    kf_true   = true_xy[kf_mask]

    fig, ax = plt.subplots(figsize=(9, 7))
    _setup_ax(ax, "Keyframe-level Gauss-Newton")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", lw=1.2, alpha=0.5, label="Ground truth (fine)")

    # Draw loop-closure edges on a fixed reference (ground truth keyframes)
    for (i, j) in lc_kf_edges:
        ax.plot([kf_true[i, 0], kf_true[j, 0]],
                [kf_true[i, 1], kf_true[j, 1]],
                color="tab:purple", lw=0.6, ls="--", alpha=0.35)

    kf_line,  = ax.plot([], [], "o-", color="tab:orange", lw=1.2,
                        ms=4, label="Keyframes (current iter)")
    lc_lines  = [ax.plot([], [], color="tab:purple", lw=0.8, ls="--", alpha=0.7)[0]
                 for _ in lc_kf_edges]
    title_txt = ax.set_title("")
    ax.legend(fontsize=8, loc="best")

    def update(frame):
        kf_xy = iters[iter_list[frame]]
        kf_line.set_data(kf_xy[:, 0], kf_xy[:, 1])
        for line, (i, j) in zip(lc_lines, lc_kf_edges):
            if i < len(kf_xy) and j < len(kf_xy):
                line.set_data([kf_xy[i, 0], kf_xy[j, 0]],
                              [kf_xy[i, 1], kf_xy[j, 1]])
        title_txt.set_text(f"Keyframe-level GN  —  iteration {iter_list[frame]}")
        return [kf_line, title_txt] + lc_lines

    # Auto-fit axes to the widest frame
    all_xy = np.vstack(list(iters.values()))
    margin = 0.5
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    anim = FuncAnimation(fig, update, frames=len(iter_list),
                         interval=120, blit=False)
    out = output
    anim.save(str(out), writer="pillow", dpi=100)
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animate", action="store_true",
                    help="Create GIF animation of keyframe GN iterations")
    ap.add_argument("--show", action="store_true",
                    help="Display plot window (in addition to saving)")
    args = ap.parse_args()

    traj_path  = OUTPUT_DIR / "hierarchical_trajectory.csv"
    edges_path = OUTPUT_DIR / "hierarchical_kf_edges.csv"
    iters_path = OUTPUT_DIR / "hierarchical_iterations.csv"

    for p in (traj_path, edges_path):
        if not p.exists():
            print(f"Missing {p} – run hierarchical_pose_graph_slam_demo first.")
            return

    true_xy, init_xy, opt_xy, kf_mask = read_trajectory(traj_path)
    odom_kf_edges, lc_kf_edges        = read_kf_edges(edges_path)

    plot_static(true_xy, init_xy, opt_xy, kf_mask,
                odom_kf_edges, lc_kf_edges,
                OUTPUT_DIR / "hierarchical_pose_graph_slam.png",
                show=args.show)

    if args.animate:
        if iters_path.exists():
            iters = read_iterations(iters_path)
            animate(true_xy, opt_xy, kf_mask,
                    odom_kf_edges, lc_kf_edges, iters,
                    OUTPUT_DIR / "hierarchical_iterations.gif",
                    show=args.show)
        else:
            print(f"No {iters_path} – skipping animation.")


if __name__ == "__main__":
    main()
