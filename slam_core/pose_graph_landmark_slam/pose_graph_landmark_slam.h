#pragma once

#include <cstddef>
#include <vector>

#include "../utils/matrix.h"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct PLPoseNode {
    double x, y, theta;
};

struct PLLandmarkNode {
    double x, y;
};

// Odometry / loop-closure edge between two robot poses (3-DOF ↔ 3-DOF)
struct PLOdometryEdge {
    size_t from, to;
    double dx, dy, dtheta;       // measured relative pose (in frame of 'from')
    kf::Matrix info;             // 3×3 information matrix
};

// Landmark observation edge from a pose to a landmark (3-DOF ↔ 2-DOF)
struct PLLandmarkEdge {
    size_t pose_idx;
    size_t landmark_idx;
    double dx, dy;               // measured relative position in robot frame
    kf::Matrix info;             // 2×2 information matrix
};

// Snapshot of the full state at one Gauss-Newton iteration
struct PLSnapshot {
    std::vector<PLPoseNode>     poses;
    std::vector<PLLandmarkNode> landmarks;
};

// ─────────────────────────────────────────────────────────────────────────────
// PoseGraphLandmarkSlam
//
// Graph-based SLAM back-end that handles both robot poses and 2-D landmarks,
// following the formulation from Stachniss – "Graph-Based SLAM with Landmarks"
// (lecture slides slam17-ls-landmarks).
//
// State vector layout  (dimension = 3·N_p + 2·N_l):
//   [x_0, y_0, θ_0,  …,  x_{N_p-1}, y_{N_p-1}, θ_{N_p-1},
//    lx_0, ly_0,  …,  lx_{N_l-1}, ly_{N_l-1}]
//
// Two edge types:
//   1. Odometry edge (pose i → pose j): 3-D error, 3×3 Jacobians — same math
//      as PoseGraphSlam.
//   2. Landmark edge (pose i → landmark l): 2-D error, 2×3 + 2×2 Jacobians.
//      Predicted observation:  z_pred = R_i^T · (lm - p_i)
//      Error:  e = z_pred − z_meas
//
// Because landmark-only constraints can leave the system rank-deficient,
// a Levenberg-Marquardt-style damping term λI is added:
//   (H + λI) · Δx = −b
//
// The first pose is anchored (large diagonal) to fix gauge freedom.
// ─────────────────────────────────────────────────────────────────────────────

class PoseGraphLandmarkSlam {
public:
    PoseGraphLandmarkSlam();

    // ── Build the graph ──────────────────────────────────────────────────────
    size_t add_pose(double x, double y, double theta);
    size_t add_landmark(double x, double y);    // initial estimate

    size_t add_odometry_edge(size_t from, size_t to,
                             double dx, double dy, double dtheta,
                             double info_pos, double info_rot);

    size_t add_landmark_edge(size_t pose_idx, size_t landmark_idx,
                             double dx, double dy,
                             double info_pos);

    // ── Optimization ─────────────────────────────────────────────────────────
    // damping: λ added to the diagonal of H before solving (H + λI)·Δx = −b
    int optimize(int max_iterations = 100, double tolerance = 1e-4,
                 double damping = 1e-4,
                 std::vector<PLSnapshot>* iter_history = nullptr);

    double total_error() const;

    // ── Accessors ────────────────────────────────────────────────────────────
    const std::vector<PLPoseNode>&      poses()      const { return poses_; }
    const std::vector<PLLandmarkNode>&  landmarks()  const { return landmarks_; }
    const std::vector<PLOdometryEdge>&  odom_edges() const { return odom_edges_; }
    const std::vector<PLLandmarkEdge>&  lm_edges()   const { return lm_edges_; }

private:
    std::vector<PLPoseNode>     poses_;
    std::vector<PLLandmarkNode> landmarks_;
    std::vector<PLOdometryEdge> odom_edges_;
    std::vector<PLLandmarkEdge> lm_edges_;

    // ── State-vector index helpers ───────────────────────────────────────────
    size_t pose_offset(size_t i) const { return 3 * i; }
    size_t lm_offset  (size_t l) const { return 3 * poses_.size() + 2 * l; }
    size_t system_dim ()         const { return 3 * poses_.size() + 2 * landmarks_.size(); }

    // ── Per-edge computations ────────────────────────────────────────────────
    void compute_odom_error(const PLOdometryEdge& e,
                            double& ex, double& ey, double& et) const;
    void compute_odom_jacobians(const PLOdometryEdge& e,
                                kf::Matrix& A, kf::Matrix& B) const;

    void compute_lm_error(const PLLandmarkEdge& e,
                          double& ex, double& ey) const;
    void compute_lm_jacobians(const PLLandmarkEdge& e,
                              kf::Matrix& A, kf::Matrix& B) const;

    static kf::Matrix cholesky_solve(const kf::Matrix& H, const kf::Matrix& b);
    static double normalize_angle(double a);
};

} // namespace slam
