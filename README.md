# SLAM Algorithms – Learning Repository

A modular, learning-focused C++/Python project for studying classic SLAM algorithms
**from first principles**. Each algorithm is implemented cleanly in isolation — no ROS,
no heavy dependencies, no magic — so you can read the math, run it, and immediately see
what it does.

The seven algorithms form a deliberate progression:
- **EKF-SLAM** — the simplest closed-form batch estimator (Kalman filter + landmarks)
- **FastSLAM 1.0** — particle filter to avoid the O(n²) covariance problem
- **GridSLAM / RBPF** — ditch landmarks entirely; build a dense occupancy grid
- **Pose Graph SLAM** — decouple front-end and back-end; solve as sparse least squares
- **Hierarchical Pose Graph** — multi-level graph for scalability and global correction
- **Pose Graph + Landmarks** — bring back explicit landmarks but keep the graph back-end
- **Robust Pose Graph** — handle wrong loop closures without any explicit rejection step

## Algorithms implemented

| # | Algorithm | C++ library | Demo | Visualization |
|---|-----------|-------------|------|---------------|
| 1 | EKF-SLAM | `slam_core/ekf_slam/` | `examples/ekf_slam_demo` | `slam_viz/plot_ekf_slam.py` |
| 2 | FastSLAM 1.0 | `slam_core/fast_slam/` | `examples/fast_slam_demo` | `slam_viz/plot_fast_slam.py` |
| 3 | GridSLAM / RBPF (FastSLAM 2.0) | `slam_core/grid_slam/` | `examples/grid_slam_demo` | `slam_viz/plot_grid_slam.py` |
| 4 | Pose Graph SLAM (Gauss-Newton) | `slam_core/pose_graph_slam/` | `examples/pose_graph_slam_demo` | `slam_viz/plot_pose_graph_slam.py` |
| 5 | Hierarchical Pose Graph SLAM | `slam_core/hierarchical_pose_graph_slam/` | `examples/hierarchical_pose_graph_slam_demo` | `slam_viz/plot_hierarchical_pose_graph_slam.py` |
| 6 | Pose Graph SLAM with Landmarks | `slam_core/pose_graph_landmark_slam/` | `examples/pose_graph_landmark_slam_demo` | `slam_viz/plot_pose_graph_landmark_slam.py` |
| 7 | Robust Pose Graph SLAM (Max-Mixture + Huber) | `slam_core/robust_pose_graph_slam/` | `examples/robust_pose_graph_slam_demo` | `slam_viz/plot_robust_pose_graph_slam.py` |

## Repository structure

```
slam_core/
  utils/                          # Matrix utilities (shared by all algorithms)
  ekf_slam/                       # EKF-SLAM static library
  fast_slam/                      # FastSLAM 1.0 static library
  grid_slam/                      # GridSLAM / RBPF static library
  pose_graph_slam/                # Pose Graph SLAM (Gauss-Newton back-end)
  hierarchical_pose_graph_slam/   # Hierarchical Pose Graph SLAM (multi-level)
  pose_graph_landmark_slam/       # Pose Graph SLAM with explicit 2-D landmarks
  robust_pose_graph_slam/         # Robust Pose Graph SLAM (Max-Mixture + Huber)
examples/
  ekf_slam_demo.cpp                       # EKF-SLAM demo runner
  fast_slam_demo.cpp                      # FastSLAM 1.0 demo runner
  grid_slam_demo.cpp                      # GridSLAM demo runner (20×20m room, 180-beam lidar)
  pose_graph_slam_demo.cpp                # Pose Graph SLAM demo (rectangular loop)
  hierarchical_pose_graph_slam_demo.cpp   # Hierarchical Pose Graph SLAM demo
  pose_graph_landmark_slam_demo.cpp       # Pose Graph + Landmarks demo
  robust_pose_graph_slam_demo.cpp         # Robust SLAM demo (injected outlier LCs)
slam_viz/
  plot_ekf_slam.py                        # EKF-SLAM visualization
  plot_fast_slam.py                       # FastSLAM 1.0 visualization
  plot_grid_slam.py                       # GridSLAM visualization (static + animation)
  plot_pose_graph_slam.py                 # Pose Graph SLAM (before/after + GN animation)
  plot_hierarchical_pose_graph_slam.py    # Hierarchical SLAM visualization
  plot_pose_graph_landmark_slam.py        # Pose Graph + Landmarks visualization
  plot_robust_pose_graph_slam.py          # Robust SLAM visualization (outlier colouring)
output/                                   # Generated CSV files and plots
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| C++17 compiler | `g++ 9+` or `clang++ 10+` |
| CMake 3.16+ | `sudo apt install cmake` |
| Python 3.8+ | For visualization scripts |
| pip packages | `matplotlib`, `numpy` — installed automatically by `run_demo.sh` |

No ROS, no Eigen, no Boost. The only C++ dependency is the custom `kf::Matrix` utility
that lives inside this repo.

---

## Learning path – suggested reading order

Work through the algorithms in order. Each one solves a specific problem left open by
the previous one.

| Step | Algorithm | Key concept to focus on | Open problem it leaves |
|------|-----------|-------------------------|------------------------|
| 1 | **EKF-SLAM** | Linearised Kalman update; augmented state vector | $O(n^2)$ cost per step; scales badly |
| 2 | **FastSLAM 1.0** | Rao-Blackwellised particles; per-landmark EKFs | Each particle carries a full map — memory grows with landmarks |
| 3 | **GridSLAM** | Occupancy grids; scan-match proposal; log-odds update | Graph memory/complexity; no compact representation of trajectory |
| 4 | **Pose Graph SLAM** | Front-end vs back-end separation; Gauss-Newton sparse solver | Monolithic graph — large environments need hierarchical structure |
| 5 | **Hierarchical Pose Graph** | Multi-level subgraph compression; local-then-global optimisation | All edges are trusted — wrong loop closures corrupt the map |
| 6 | **Pose Graph + Landmarks** | Mixed state (poses + 2-D points) in one graph; damping | Can we handle noisy/wrong data associations without outlier rejection? |
| 7 | **Robust Pose Graph** | Max-Mixture edge model; Huber kernel M-estimator | — (this is the current end of the progression) |

**Where to start reading code:**
- Read `slam_core/<algo>/<algo>.h` first — it documents the public API and data structures.
- Then read the matching `_demo.cpp` in `examples/` to see the full scenario setup.
- Finally read `<algo>.cpp` for the implementation details.

---

## Code architecture

### Common pattern across all algorithms

Every algorithm follows the same three-layer structure:

```
slam_core/<algo>/
  <algo>.h          ← class declaration, all data structures defined here
  <algo>.cpp        ← implementation
examples/
  <algo>_demo.cpp   ← scenario setup, calls the SLAM class, writes CSV to output/
slam_viz/
  plot_<algo>.py    ← reads those CSVs, produces static plot or animation
```

`run_demo.sh` chains all three: build → run demo → run viz.

### Data flow

```
C++ demo  →  output/*.csv  →  Python viz script  →  plot window / file
```

The CSV files are intentionally human-readable so you can inspect them directly
(e.g., `cat output/pose_graph_trajectory.csv | head -20`) to understand what
the algorithm produces before looking at the plots.

### Shared matrix utility (`slam_core/utils/`)

All algorithms use `kf::Matrix` — a minimal dense matrix class written from scratch
so the math stays fully visible. Operations available:

| Method | Description |
|--------|-------------|
| arithmetic (+, -, *) | Standard matrix/scalar ops |
| `transpose()` | Returns Mᵀ |
| `inverse()` | Gauss-Jordan inverse |
| `det()` | General N×N determinant via LU decomposition with partial pivoting |

If you are used to Eigen, `kf::Matrix` is the same idea but deliberately unoptimised —
every operation is a plain loop so you can read what it does.

---

## Quick start – one command

```bash
# EKF-SLAM (default)
./run_demo.sh --show

# FastSLAM 1.0
./run_demo.sh --algo fast_slam --show

# GridSLAM / RBPF
./run_demo.sh --algo grid_slam --show

# Pose Graph SLAM (Gauss-Newton back-end)
./run_demo.sh --algo pose_graph_slam --show

# Hierarchical Pose Graph SLAM
./run_demo.sh --algo hierarchical_pose_graph_slam --show

# Pose Graph SLAM with Landmarks
./run_demo.sh --algo pose_graph_landmark_slam --show

# Robust Pose Graph SLAM (Max-Mixture + Huber kernel)
./run_demo.sh --algo robust_pose_graph_slam --show

# With animation (shows optimisation converging)
./run_demo.sh --algo pose_graph_slam --animate --show

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
| `--algo ALGO` | `ekf_slam`, `fast_slam`, `grid_slam`, `pose_graph_slam`, `hierarchical_pose_graph_slam`, `pose_graph_landmark_slam`, `robust_pose_graph_slam` | `ekf_slam` |
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
./build/examples/pose_graph_slam_demo
./build/examples/hierarchical_pose_graph_slam_demo
./build/examples/pose_graph_landmark_slam_demo
./build/examples/robust_pose_graph_slam_demo
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

### Pose Graph SLAM (Gauss-Newton back-end)

Implements the **graph-based SLAM back-end** from the Burgard/Stachniss lecture.

**Key idea:** Represent the SLAM problem as a **pose graph**:
- **Nodes** = robot poses at each timestep
- **Edges** = spatial constraints (odometry between consecutive poses, loop closures
  between revisited poses)
- **Optimisation** = Gauss-Newton least squares minimising constraint errors

**Algorithm (per Gauss-Newton iteration):**
1. For each edge (i,j), compute the **error** between predicted and measured relative
   pose: $e_{ij} = R_i^T(p_j - p_i) - z_{ij}$
2. Compute **Jacobians** $A_{ij} = \partial e / \partial x_i$ and $B_{ij} = \partial e / \partial x_j$
3. Build the **information system**: $H = \sum A^T \Omega A$, $b = \sum A^T \Omega e$
4. **Anchor** the first node (add large value to $H_{00}$) to fix gauge freedom
5. Solve $H \cdot \Delta x = -b$ via **Cholesky decomposition**
6. Update all poses: $x \leftarrow x + \Delta x$
7. Repeat until convergence ($\|\Delta x\| < \varepsilon$)

**Edge types:**

| Type | Source | Information |
|------|--------|-------------|
| Odometry | Sequential motion | $\Omega_{\text{odom}} = \text{diag}(1/\sigma_{\text{lin}}^2, 1/\sigma_{\text{lin}}^2, 1/\sigma_{\text{ang}}^2)$ |
| Loop closure | Revisiting same area (simulated scan-match) | $\Omega_{\text{lc}}$ (similar, different noise level) |

**Demo scenario:**
- Robot drives two laps around a 10×7 m rectangle (~136 poses)
- Odometry noise: 3 cm translation, 0.6° rotation per step
- Loop closures detected when true poses are within 1.5 m (min 20-step gap)
- Dramatic drift visible before optimisation; clean correction after

**Output files:**

| File | Contents |
|------|----------|
| `output/pose_graph_trajectory.csv` | Per-node: true, initial (odom), optimised poses |
| `output/pose_graph_edges.csv` | All edges with type (odometry / loop_closure) |
| `output/pose_graph_iterations.csv` | Node positions at each GN iteration (for animation) |

### Pose Graph SLAM with Landmarks

Extends the standard pose graph back-end to include **explicit 2-D landmark nodes**
alongside pose nodes — the kind of map used in EKF-SLAM but solved via graph
optimisation instead of a Kalman filter.

**State layout:** `[x₀,y₀,θ₀, …, xₙ,yₙ,θₙ | lx₀,ly₀, …, lxₘ,lyₘ]`

**Two edge types:**

| Type | Connects | Measurement | Jacobian sizes |
|------|----------|-------------|----------------|
| Odometry | pose i → pose j | relative pose (dx, dy, dθ) | A(3×3), B(3×3) |
| Landmark | pose i → landmark l | relative position (dx, dy) in body frame | A(2×3), B(2×2) |

**Landmark error:** $e = R_i^T(l - p_i) - z$ where $R_i^T$ rotates world-frame
displacement into the robot body frame at pose i.

**Damping:** a small $\lambda I$ is added to the diagonal of H before solving to
handle rank deficiency when landmarks are only observed from a few poses.

**Demo scenario:**
- Same 10×7 m rectangular trajectory (137 poses, two laps)
- 12 landmarks scattered in the environment
- 674 landmark observation edges + 136 odometry edges
- Converges in ~5 iterations; error reduced from ~85 000 → ~1 300

**Output files:**

| File | Contents |
|------|----------|
| `output/pg_lm_trajectory.csv` | Per-node: true, initial, optimised poses |
| `output/pg_lm_landmarks.csv` | True and optimised landmark positions |
| `output/pg_lm_edges.csv` | All edges with type |
| `output/pg_lm_iterations.csv` | Pose + landmark snapshots per GN iteration |

### Robust Pose Graph SLAM (Max-Mixture + Huber kernel)

Adds **two robustness mechanisms** on top of the standard Gauss-Newton pose graph
back-end to handle wrong data associations (outlier loop closures):

#### 1. Max-Mixture (MM)

Each edge stores a `vector<MMComponent>` — a set of Gaussian hypotheses.
Before building the linear system each iteration, the component with the
highest log-likelihood given the current state is selected:

$$k^* = \arg\max_k \left[ \log w_k + \tfrac{1}{2}\log|\Omega_k| - \tfrac{3}{2}\log 2\pi - \tfrac{1}{2}\chi^2_k \right]$$

For loop-closure edges, two components are supplied:
- **Component 0 (inlier):** tight Gaussian around scan-match measurement, weight = 0.9
- **Component 1 (null):** near-zero information (flat prior), weight = 0.1

When a loop closure is correct the inlier component wins. When it is wrong (large
error → huge $-\tfrac{1}{2}\chi^2$), the null component wins, its near-zero $\Omega$
making the edge contribute essentially nothing to H and b — the outlier is suppressed
without any explicit rejection step.

#### 2. Robust Kernel (Huber)

After component selection, the selected $\Omega_{k^*}$ is further scaled:

$$\Omega_{\text{eff}} = w(\chi^2) \cdot \Omega_{k^*}, \quad w = \begin{cases} 1 & \chi^2 \le \delta^2 \\ \delta / \sqrt{\chi^2} & \chi^2 > \delta^2 \end{cases}$$

This down-weights any remaining high-error inlier-classified edges linearly
(quadratic cost → linear cost) — the industry-standard M-estimator.
Cauchy kernel ($w = 1/(1 + \chi^2/\delta^2)$) is also available.

**`matrix.h` addition:** `Matrix::det()` — general N×N determinant via LU
decomposition with partial pivoting (needed for the $\log|\Omega|$ normalisation term).

**Demo scenario:**
- Same 10×7 m rectangle × 2 trajectory (137 poses)
- 381 true loop-closure MM edges + 3 injected false loop closures
- All 3 false LCs correctly suppressed (null component wins)
- 381 true LCs correctly classified as inlier
- Converges in ~13 iterations

**Output files:**

| File | Contents |
|------|----------|
| `output/robust_trajectory.csv` | Per-node: true, initial, optimised poses |
| `output/robust_edges.csv` | Edges with type, injected-outlier flag, MM component selected |
| `output/robust_iterations.csv` | Node positions at each GN iteration |

**Visualization edge colours:**
- Grey — odometry
- Steel blue — loop closure, inlier component selected
- Crimson dashed — loop closure, null component selected (suppressed)
- Orange-red dashed — injected false outlier (correctly suppressed)

## Notes and design decisions

- **Intentionally no ROS / Eigen / Ceres.** Everything is implemented from scratch so
  the math in the algorithm notes above maps 1-to-1 to the code. No black boxes.
- **Known data association** in EKF-SLAM and FastSLAM 1.0. Landmark IDs are given to
  the algorithm rather than computed — so you can focus on the estimator, not matching.
- **Simulated front-end** in all pose graph variants. Loop closures are detected using
  ground-truth pose proximity to avoid front-end complexity obscuring back-end learning.
- **Gauss-Newton, not g2o / iSAM.** The Cholesky solver is written by hand on `kf::Matrix`.
  It is not sparse — it is $O(n^3)$ — but for ≤ 200 nodes this is instant and fully
  readable. A production system would use `g2o` or `GTSAM` with sparse Cholesky.
- **Max-Mixture, not switchable constraints / DCS.** The MM formulation (Olson & Agarwal
  2012) is the cleanest way to understand robust estimation in a single pass without
  modifying the graph structure.
- **The matrix library is the floor, not the ceiling.** Once you understand what each
  algorithm is doing, replace `kf::Matrix` with Eigen and the structure/interfaces stay
  identical — that is a good next exercise.
