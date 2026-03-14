# SLAM Algorithms – Learning Repository

A modular, learning-focused C++/Python project for exploring classic SLAM algorithms.
Each algorithm lives in its own library under `slam_core/`, with a standalone demo in
`examples/` and a visualization script in `slam_viz/`.

## Algorithms implemented

| Algorithm              | C++ library               | Demo                          | Visualization                    |
|------------------------|---------------------------|-------------------------------|----------------------------------|
| EKF-SLAM               | `slam_core/ekf_slam/`     | `examples/ekf_slam_demo`      | `slam_viz/plot_ekf_slam.py`      |
| FastSLAM 1.0           | `slam_core/fast_slam/`    | `examples/fast_slam_demo`     | `slam_viz/plot_fast_slam.py`     |
| GridSLAM / RBPF (FastSLAM 2.0) | `slam_core/grid_slam/` | `examples/grid_slam_demo` | `slam_viz/plot_grid_slam.py` |

## Repository structure

```
slam_core/
  utils/                 # Matrix utilities (shared by all algorithms)
  ekf_slam/              # EKF-SLAM static library
  fast_slam/             # FastSLAM 1.0 static library
  grid_slam/             # GridSLAM / RBPF static library
examples/
  ekf_slam_demo.cpp      # EKF-SLAM demo runner
  fast_slam_demo.cpp     # FastSLAM 1.0 demo runner
  grid_slam_demo.cpp     # GridSLAM demo runner (20×20m room, 180-beam lidar)
slam_viz/
  plot_ekf_slam.py       # EKF-SLAM visualization
  plot_fast_slam.py      # FastSLAM 1.0 visualization
  plot_grid_slam.py      # GridSLAM visualization (static + animated map building)
output/                  # Generated CSV files and plots
```

## Quick start – one command

```bash
# EKF-SLAM (default)
./run_demo.sh --show

# FastSLAM 1.0
./run_demo.sh --algo fast_slam --show

# GridSLAM / RBPF
./run_demo.sh --algo grid_slam --show

# With animation (shows incremental map building)
./run_demo.sh --algo grid_slam --animate --show

# More timesteps
./run_demo.sh --algo ekf_slam --steps 500 --show
```

`run_demo.sh` will:
1. Build all demos with CMake.
2. Run the selected C++ demo (writes CSV output under `output/`).
3. Set up a Python venv and install requirements if needed.
4. Run the matching visualization script.

### `run_demo.sh` options

| Flag | Description | Default |
|------|-------------|---------|
| `--algo ALGO` | `ekf_slam`, `fast_slam`, or `grid_slam` | `ekf_slam` |
| `--steps N` | Number of simulation timesteps | `250` |
| `--animate` | Animated visualization instead of static PNG | off |
| `--show` | Open plot window interactively | off |

## Manual build

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run a demo from the repo root (output written to `./output/`):

```bash
./build/examples/ekf_slam_demo --steps 300
./build/examples/fast_slam_demo --steps 300
./build/examples/grid_slam_demo --steps 400
```

## Algorithm notes

### EKF-SLAM

- Full-state vector `[x, y, θ, lx₀, ly₀, …, lxₙ, lyₙ]` augmented with all landmark positions.
- Motion model: nonlinear unicycle; covariance propagated via linearised Jacobian.
- Measurement update: range-bearing, standard EKF innovation update.
- Landmark initialisation from first observation; covariance grows until observed.

### FastSLAM 1.0

- Particle filter where **each particle** carries its own pose estimate + **one EKF per landmark**.
- Predict: sample a new pose for every particle from the noisy motion model.
- Update: for each particle and each observation, run the per-landmark 2×2 EKF update and weight the particle by the observation likelihood.
- Resample: low-variance (systematic) resampling after every timestep.
- Output: best-particle pose, weighted-mean landmark estimates, and a particle-cloud CSV for visualisation.

### GridSLAM (RBPF / FastSLAM 2.0 with Occupancy Grids)

Implements a **Rao-Blackwellised Particle Filter (RBPF)** where each particle maintains
its own full occupancy grid map of the environment — the approach used by the real-world
[GMapping](https://openslam-org.github.io/gmapping.html) algorithm.

**Key idea:** Instead of tracking discrete landmarks, the robot builds a dense 2-D
occupancy grid. Each particle carries a different map hypothesis, and particles with
maps that better explain the laser scans accumulate higher weight and survive resampling.

**Per-timestep RBPF steps:**

1. **Predict** — sample a noisy pose from the velocity motion model for each particle.
2. **Scan-match** (FastSLAM 2.0 improvement) — search a small 7×7×7 window around the
   predicted pose, scoring each candidate pose against the particle's current map using
   `scan_score()`. Accept the highest-scoring pose as the proposal.
3. **Weight** — set particle weight to `exp(best_score / num_valid_beams)`.
4. **Normalise** — divide all weights by their sum.
5. **Integrate scan** — update each particle's occupancy grid via Bresenham ray-tracing
   (free cells along the ray, occupied cell at the endpoint).
6. **Resample** — low-variance (systematic) resampling; high-weight particles are
   duplicated, low-weight particles are discarded.

**Why scan-matching matters (FastSLAM 1.0 vs 2.0):**

| | FastSLAM 1.0 | FastSLAM 2.0 / GridSLAM |
|--|--|--|
| Proposal | Motion model only | Motion model + scan-match refinement |
| Pose quality | Drifts with noise | Corrected against current map |
| Convergence | Slower, more particles needed | Faster, fewer particles needed |

**Sensor model:** Log-odds inverse sensor model.
- Ray endpoint (hit): `log_odds += lo_occ` (default +0.65)
- Ray free cells: `log_odds += lo_free` (default −0.40)
- Clipped to `[lo_min, lo_max]` = `[−5.0, +5.0]`

**Coordinate frames:**
- `(x, y, θ)` — world frame (fixed, metres + radians)
- `scan.ranges[i]` — robot frame (distance from robot centre)
- Grid cells `(col, row)` — grid frame (discrete, world-aligned)

**Demo scenario:**
- 20×20 m room with 4 outer walls + 4 internal obstacles
- 180-beam 360° lidar, 12 m range, 5 cm range noise
- 30 particles, 0.2 m/cell resolution, ±13 m grid
- Robot follows a coverage trajectory through hand-picked open-space waypoints
  with line-of-sight checks to avoid wall-crossing

**Output files:**

| File | Contents |
|------|----------|
| `output/grid_slam_trajectory.csv` | True pose + best-particle estimated pose per step |
| `output/grid_slam_particles.csv` | Full particle cloud (x, y, θ, weight) per step |
| `output/grid_slam_map.csv` | Final best-particle SLAM occupancy map |
| `output/grid_slam_map_snapshots.csv` | Incremental map snapshots every 20 steps (for animation) |
| `output/grid_slam_gt_map.csv` | Ground-truth room geometry rasterised to grid |

## Notes

- Designed as a learning baseline, not a production system.
- EKF-SLAM and FastSLAM 1.0 use known data association (landmark IDs provided).
- GridSLAM uses dense occupancy grids — no explicit landmarks or data association needed.
- The matrix library (`slam_core/utils/`) is intentionally minimal to keep the math transparent.
