#pragma once

#include <cstddef>
#include <vector>

#include "../utils/matrix.h"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct PoseNode {
    double x, y, theta;
};

struct PoseEdge {
    size_t from, to;
    double dx, dy, dtheta;     // measured relative pose (in frame of 'from')
    kf::Matrix info;           // 3×3 information matrix (inverse covariance)
};

// ─────────────────────────────────────────────────────────────────────────────
// PoseGraphSlam
//
// Graph-based SLAM back-end (from the Burgard/Stachniss lecture).
//
// 1. Build a pose graph: nodes = robot poses, edges = spatial constraints
//    (odometry between consecutive poses, loop closures between revisited
//    poses).
// 2. Optimize all node positions jointly via Gauss-Newton least squares to
//    minimize the total constraint error.
//
// The error for edge (i,j) with measurement z_ij:
//     e_ij = R_i^T * (p_j - p_i)  −  z_ij      (position part)
//     e_ij_theta = theta_j − theta_i − z_dtheta  (orientation part)
//
// Gauss-Newton iterates:  H * dx = −b   where
//     H = Σ J_ij^T  Ω_ij  J_ij       (information / Hessian)
//     b = Σ J_ij^T  Ω_ij  e_ij       (gradient)
// First node is anchored to remove gauge freedom.
// ─────────────────────────────────────────────────────────────────────────────

class PoseGraphSlam {
public:
    PoseGraphSlam();

    // ── Build the graph ──────────────────────────────────────────────────────
    size_t add_node(double x, double y, double theta);

    // Returns the index of the newly created edge.
    size_t add_edge(size_t from, size_t to,
                    double dx, double dy, double dtheta,
                    double info_pos, double info_rot);

    // Replace the information matrix of an existing edge in-place.
    // edge_idx is the value returned by add_edge().
    void update_edge_info(size_t edge_idx, double info_pos, double info_rot);

    // ── Optimization ─────────────────────────────────────────────────────────
    // Returns the number of iterations performed.
    // Stores per-iteration node snapshots when iter_history is non-null.
    int optimize(int max_iterations = 100, double tolerance = 1e-4,
                 std::vector<std::vector<PoseNode>>* iter_history = nullptr);

    double total_error() const;

    // Assemble and return the Hessian H at the current linearisation point
    // WITHOUT anchor augmentation.  Useful for Schur complement computations
    // (e.g., hierarchical SLAM information marginalisation).
    kf::Matrix assemble_hessian() const;

    // Marginal 3×3 covariance for each node: diagonal blocks of (H+anchor)^{-1}.
    // The anchor keeps the first node fixed for numerical stability.
    // An empty vector is returned if there are no nodes.
    std::vector<kf::Matrix> marginal_covariances() const;

    // ── Accessors ────────────────────────────────────────────────────────────
    const std::vector<PoseNode>& nodes() const { return nodes_; }
    const std::vector<PoseEdge>& edges() const { return edges_; }

private:
    std::vector<PoseNode> nodes_;
    std::vector<PoseEdge> edges_;

    void compute_error(const PoseEdge& edge,
                       double& ex, double& ey, double& et) const;

    void compute_jacobians(const PoseEdge& edge,
                           kf::Matrix& A, kf::Matrix& B) const;

    static kf::Matrix cholesky_solve(const kf::Matrix& H, const kf::Matrix& b);

    static double normalize_angle(double a);
};

} // namespace slam
