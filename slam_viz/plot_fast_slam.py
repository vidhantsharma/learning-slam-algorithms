import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation


# ─────────────────────────────────────────────────────────────────────────────
# Data readers
# ─────────────────────────────────────────────────────────────────────────────

def read_trajectory(path: Path):
    steps, times, true_xy, est_xy, headings = [], [], [], [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["step"]))
            times.append(float(row["t"]))
            true_xy.append((float(row["x_true"]), float(row["y_true"])))
            est_xy.append((float(row["x_est"]), float(row["y_est"])))
            headings.append(float(row["theta_est"]))
    return (
        np.array(steps),
        np.array(times),
        np.array(true_xy),
        np.array(est_xy),
        np.array(headings),
    )


def read_landmarks(path: Path):
    true_lm, est_lm = [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            true_lm.append((float(row["x_true"]), float(row["y_true"])))
            est_lm.append((float(row["x_est"]), float(row["y_est"])))
    return np.array(true_lm), np.array(est_lm)


def read_landmark_estimates(path: Path):
    """Returns {step: [(id, x_est, y_est, cov_xx, cov_xy, cov_yy), …]}."""
    by_step = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            entry = (
                int(row["id"]),
                float(row["x_est"]),
                float(row["y_est"]),
                float(row["cov_xx"]),
                float(row["cov_xy"]),
                float(row["cov_yy"]),
            )
            by_step.setdefault(step, []).append(entry)
    return by_step


def read_particles(path: Path):
    """Returns {step: [(x, y, theta, weight), …]}."""
    by_step = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            entry = (
                float(row["x"]),
                float(row["y"]),
                float(row["theta"]),
                float(row["weight"]),
            )
            by_step.setdefault(step, []).append(entry)
    return by_step


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def covariance_to_ellipse(cov_xx, cov_xy, cov_yy, n_std: float = 2.0):
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0.0))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


def _nearest_logged_step(step, by_step):
    """Return the key in by_step that is closest to (but ≤) step."""
    keys = sorted(k for k in by_step if k <= step)
    return keys[-1] if keys else None


# ─────────────────────────────────────────────────────────────────────────────
# Static plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_static(true_xy, est_xy, headings, true_lm, est_lm,
                lm_by_step, particles_by_step, output: Path, n_std: float):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(true_xy[:, 0], true_xy[:, 1], "k--", label="True path")
    ax.plot(est_xy[:, 0], est_xy[:, 1], "b",   label="FastSLAM estimate (best particle)")

    if len(true_lm) > 0:
        ax.scatter(true_lm[:, 0], true_lm[:, 1], c="green", marker="^",
                   zorder=5, label="True landmarks")
    if len(est_lm) > 0:
        ax.scatter(est_lm[:, 0], est_lm[:, 1], c="red", marker="x",
                   zorder=5, label="Estimated landmarks (weighted mean)")

    # Particle cloud at final step
    last_step = len(est_xy) - 1
    pkey = _nearest_logged_step(last_step, particles_by_step)
    if pkey is not None:
        pts = np.array([(px, py) for px, py, _, _ in particles_by_step[pkey]])
        ax.scatter(pts[:, 0], pts[:, 1], c="gray", s=8, alpha=0.4,
                   label="Particles (final)")

    # Covariance ellipses for estimated landmarks from the best particle at every step
    for step in range(len(est_xy)):
        # Find best particle (highest weight) for this step
        pkey = _nearest_logged_step(step, particles_by_step)
        if pkey is not None:
            particles = particles_by_step[pkey]
            best_idx = max(range(len(particles)), key=lambda i: particles[i][3])
            # Only show ellipses for the best particle's landmarks
            ekey = _nearest_logged_step(step, lm_by_step)
            if ekey is not None:
                # Assuming landmark estimates are ordered by particle index, select only those for best_idx
                # If not, you may need to adjust how landmark estimates are stored
                for entry in lm_by_step[ekey]:
                    # If your CSV stores per-particle, filter by particle index here
                    # Otherwise, just plot all (as before)
                    _, x_est, y_est, cov_xx, cov_xy, cov_yy = entry
                    w, h, angle = covariance_to_ellipse(cov_xx, cov_xy, cov_yy, n_std)
                    ellipse = patches.Ellipse((x_est, y_est), w, h, angle=angle,
                                              edgecolor="red", facecolor="none", alpha=0.6)
                    ax.add_patch(ellipse)
                    ax.scatter(x_est, y_est, c="red", marker="x", zorder=5)

    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title("FastSLAM 1.0 – 2D Scenario (final frame)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved plot to {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

def animate_results(steps, true_xy, est_xy, headings, true_lm,
                    lm_by_step, particles_by_step, n_std: float):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    all_points = np.vstack([true_xy, est_xy, true_lm]) if len(true_lm) else np.vstack([true_xy, est_xy])
    min_xy = all_points.min(axis=0) - 2.0
    max_xy = all_points.max(axis=0) + 2.0

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.grid(True)

        ax.plot(true_xy[: frame_idx + 1, 0], true_xy[: frame_idx + 1, 1],
                "k--", label="True path")
        ax.plot(est_xy[: frame_idx + 1, 0], est_xy[: frame_idx + 1, 1],
                "b",   label="FastSLAM estimate")

        if len(true_lm) > 0:
            ax.scatter(true_lm[:, 0], true_lm[:, 1], c="green", marker="^",
                       zorder=5, label="True landmarks")

        step = steps[frame_idx]

        # Particle cloud
        pkey = _nearest_logged_step(step, particles_by_step)
        if pkey is not None:
            pts = np.array([(px, py) for px, py, _, _ in particles_by_step[pkey]])
            ax.scatter(pts[:, 0], pts[:, 1], c="gray", s=8, alpha=0.35,
                       label="Particles")

        # Landmark EKFs: show covariance ellipses and crosses for each landmark from best particle at current step
        ekey = _nearest_logged_step(steps[frame_idx], lm_by_step)
        if ekey is not None:
            for entry in lm_by_step[ekey]:
                _, x_est, y_est, cov_xx, cov_xy, cov_yy = entry
                # Scale covariance by 100 for visualization
                w, h, angle = covariance_to_ellipse(cov_xx * 100, cov_xy * 100, cov_yy * 100, n_std)
                ellipse = patches.Ellipse((x_est, y_est), w, h, angle=angle,
                                          edgecolor="red", facecolor="none", alpha=0.6)
                ax.add_patch(ellipse)
                ax.scatter(x_est, y_est, c="red", marker="x", zorder=5)

        # Robot arrow (best particle)
        robot_x, robot_y = est_xy[frame_idx]
        theta = headings[frame_idx]
        ax.arrow(robot_x, robot_y,
                 0.5 * np.cos(theta), 0.5 * np.sin(theta),
                 head_width=0.2, head_length=0.25, color="blue", alpha=0.8)

        ax.set_title(f"FastSLAM 1.0 – 2D Scenario (step {step})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")

    return FuncAnimation(fig, draw_frame, frames=len(steps), interval=60, repeat=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot FastSLAM 1.0 2D scenario output")
    parser.add_argument("--trajectory",         type=Path, default=Path("output/fast_slam_trajectory.csv"))
    parser.add_argument("--landmarks",          type=Path, default=Path("output/fast_slam_landmarks.csv"))
    parser.add_argument("--landmark-estimates", type=Path, default=Path("output/fast_slam_landmark_estimates.csv"))
    parser.add_argument("--particles",          type=Path, default=Path("output/fast_slam_particles.csv"))
    parser.add_argument("--output",             type=Path, default=Path("output/fast_slam_plot.png"))
    parser.add_argument("--animate",  action="store_true")
    parser.add_argument("--show",     action="store_true")
    parser.add_argument("--n-std",    type=float, default=2.0)
    args = parser.parse_args()

    steps, times, true_xy, est_xy, headings = read_trajectory(args.trajectory)
    true_lm, est_lm                          = read_landmarks(args.landmarks)
    lm_by_step                               = read_landmark_estimates(args.landmark_estimates)
    particles_by_step                        = read_particles(args.particles)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.animate:
        animation = animate_results(steps, true_xy, est_xy, headings, true_lm,
                                    lm_by_step, particles_by_step, args.n_std)
        if args.show:
            plt.show()
        else:
            animation.save(str(args.output.with_suffix(".gif")), writer="pillow", fps=20)
            print(f"Saved animation to {args.output.with_suffix('.gif')}")
    else:
        plot_static(true_xy, est_xy, headings, true_lm, est_lm,
                    lm_by_step, particles_by_step, args.output, args.n_std)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
