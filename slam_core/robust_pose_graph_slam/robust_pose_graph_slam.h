#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../utils/matrix.h"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct RobPoseNode {
    double x, y, theta;
};

// One Gaussian component of a Max-Mixture edge.
// A single-Gaussian edge just has one component with log_weight = 0.
struct MMComponent {
    double dx, dy, dtheta;   // measurement for this mode
    kf::Matrix info;          // 3×3 information matrix
    double log_weight;        // log(mixture weight), e.g. log(0.9)
};

struct RobustPoseEdge {
    size_t from, to;
    std::vector<MMComponent> components;  // ≥ 1
    std::string tag;                      // "odometry" or "loop_closure"
};

// ─────────────────────────────────────────────────────────────────────────────
// Robust kernel selection
//
// None   – plain Gauss-Newton (no down-weighting)
// Huber  – quadratic for inliers, linear for outliers (industry default)
// Cauchy – continuously decaying weight; heavier tails than Huber
// ─────────────────────────────────────────────────────────────────────────────

enum class RobustKernel { None, Huber, Cauchy };

// ─────────────────────────────────────────────────────────────────────────────
// RobustPoseGraphSlam
//
// Extends the standard Gauss-Newton pose-graph back-end with two mechanisms
// from the Stachniss / Chebrolu lecture on Robust SLAM:
//
// 1. Max-Mixture (MM) — each edge can carry multiple Gaussian hypotheses.
//    Before building the linear system, the component with the highest
//    log-likelihood given the current state is selected and used as a single
//    Gaussian.  A "null" (flat) component with low mixture weight serves as
//    the outlier hypothesis: when the error under all inlier hypotheses is
//    large the null component wins, effectively zeroing the edge contribution.
//
// 2. Robust kernel weighting — the selected component's information matrix Ω
//    is scaled by a per-edge weight w(χ²):
//        Huber:  w = 1                    if χ² ≤ δ²
//                w = δ / √χ²             if χ² > δ²
//        Cauchy: w = 1 / (1 + χ²/δ²)
//    The weighted information Ω_eff = w · Ω replaces Ω everywhere in H & b.
//
// Usage:
//   RobustPoseGraphSlam rpg(RobustKernel::Huber, /*delta=*/1.0);
//   rpg.add_node(...)
//   rpg.add_edge(...)        // single-Gaussian edge (e.g. odometry)
//   rpg.add_mm_edge(...)     // Max-Mixture edge (e.g. ambiguous loop closure)
//   rpg.optimize(...)
// ─────────────────────────────────────────────────────────────────────────────

class RobustPoseGraphSlam {
public:
    explicit RobustPoseGraphSlam(RobustKernel kernel = RobustKernel::Huber,
                                  double kernel_delta = 1.0);

    // ── Graph building ───────────────────────────────────────────────────────
    size_t add_node(double x, double y, double theta);

    // Standard single-Gaussian edge (Huber/Cauchy kernel still applies).
    size_t add_edge(size_t from, size_t to,
                    double dx, double dy, double dtheta,
                    double info_pos, double info_rot,
                    const std::string& tag = "odometry");

    // Max-Mixture edge: caller provides all components.
    // Typical usage: two components — one for the inlier hypothesis and one
    // diffuse "null" component for the outlier hypothesis.
    size_t add_mm_edge(size_t from, size_t to,
                       std::vector<MMComponent> components,
                       const std::string& tag = "loop_closure");

    // ── Optimization ─────────────────────────────────────────────────────────
    int optimize(int max_iterations = 100, double tolerance = 1e-4,
                 std::vector<std::vector<RobPoseNode>>* iter_history = nullptr);

    double total_error() const;

    // ── Accessors ────────────────────────────────────────────────────────────
    const std::vector<RobPoseNode>&     nodes() const { return nodes_; }
    const std::vector<RobustPoseEdge>&  edges() const { return edges_; }

    // Index of the winning MM component per edge, updated after optimize().
    // 0 = first (inlier) component won; 1 = null component won (outlier).
    const std::vector<int>& selected_components() const { return sel_; }

private:
    std::vector<RobPoseNode>    nodes_;
    std::vector<RobustPoseEdge> edges_;
    RobustKernel                kernel_;
    double                      delta_;
    std::vector<int>            sel_;  // per-edge selected component index

    // ── Per-edge helpers ─────────────────────────────────────────────────────

    // Compute the 3-D pose error using a specific measurement component.
    void compute_error(const RobustPoseEdge& edge, const MMComponent& comp,
                       double& ex, double& ey, double& et) const;

    // Jacobians of the error w.r.t. nodes — identical to standard pose graph.
    // A = ∂e/∂[x_i, y_i, θ_i]  (3×3)
    // B = ∂e/∂[x_j, y_j, θ_j]  (3×3)
    void compute_jacobians(const RobustPoseEdge& edge,
                           kf::Matrix& A, kf::Matrix& B) const;

    // Select the Max-Mixture component with the highest log-likelihood.
    // Returns the component index; also fills log_ll with that log-likelihood.
    int select_component(const RobustPoseEdge& edge, double& log_ll) const;

    // Log-likelihood of one component given the current node estimates.
    // ll = log_weight + 0.5*log|Ω| - 1.5*log(2π) - 0.5*χ²
    double component_log_likelihood(const RobustPoseEdge& edge,
                                    const MMComponent& comp) const;

    // Robust kernel weight  w(χ²):  Om_eff = w · Ω_component
    double kernel_weight(double chi2) const;

    static kf::Matrix cholesky_solve(const kf::Matrix& H, const kf::Matrix& b);
    static double normalize_angle(double a);
};

} // namespace slam
