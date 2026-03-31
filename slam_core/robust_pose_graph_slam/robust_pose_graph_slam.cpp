#include "robust_pose_graph_slam.h"

#include <cmath>
#include <limits>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

RobustPoseGraphSlam::RobustPoseGraphSlam(RobustKernel kernel, double kernel_delta)
    : kernel_(kernel), delta_(kernel_delta) {}

// ─────────────────────────────────────────────────────────────────────────────
// Graph building
// ─────────────────────────────────────────────────────────────────────────────

size_t RobustPoseGraphSlam::add_node(double x, double y, double theta) {
    nodes_.push_back({x, y, theta});
    return nodes_.size() - 1;
}

size_t RobustPoseGraphSlam::add_edge(size_t from, size_t to,
                                      double dx, double dy, double dtheta,
                                      double info_pos, double info_rot,
                                      const std::string& tag) {
    kf::Matrix info(3, 3, 0.0);
    info(0, 0) = info_pos;
    info(1, 1) = info_pos;
    info(2, 2) = info_rot;

    MMComponent comp;
    comp.dx         = dx;
    comp.dy         = dy;
    comp.dtheta     = dtheta;
    comp.info       = info;
    comp.log_weight = 0.0;   // log(1) — single component, full weight

    edges_.push_back({from, to, {comp}, tag});
    sel_.push_back(0);
    return edges_.size() - 1;
}

size_t RobustPoseGraphSlam::add_mm_edge(size_t from, size_t to,
                                         std::vector<MMComponent> components,
                                         const std::string& tag) {
    edges_.push_back({from, to, std::move(components), tag});
    sel_.push_back(0);
    return edges_.size() - 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_error – mismatch between predicted and measured relative pose
//
// Predicted relative pose from i to j (in frame i):
//     dx_pred = cos(θ_i)*(x_j − x_i) + sin(θ_i)*(y_j − y_i)
//     dy_pred = −sin(θ_i)*(x_j − x_i) + cos(θ_i)*(y_j − y_i)
//     dθ_pred = θ_j − θ_i
// Error = predicted − measured (from the selected component)
// ─────────────────────────────────────────────────────────────────────────────

void RobustPoseGraphSlam::compute_error(const RobustPoseEdge& edge,
                                         const MMComponent& comp,
                                         double& ex, double& ey,
                                         double& et) const {
    const RobPoseNode& ni = nodes_[edge.from];
    const RobPoseNode& nj = nodes_[edge.to];

    double dx = nj.x - ni.x;
    double dy = nj.y - ni.y;
    double ct = std::cos(ni.theta);
    double st = std::sin(ni.theta);

    double dx_pred =  ct * dx + st * dy;
    double dy_pred = -st * dx + ct * dy;
    double dt_pred = nj.theta - ni.theta;

    ex = dx_pred - comp.dx;
    ey = dy_pred - comp.dy;
    et = normalize_angle(dt_pred - comp.dtheta);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_jacobians – same as PoseGraphSlam (measurement-independent)
//
// A = ∂e/∂[x_i, y_i, θ_i]  (3×3)
// B = ∂e/∂[x_j, y_j, θ_j]  (3×3)
// ─────────────────────────────────────────────────────────────────────────────

void RobustPoseGraphSlam::compute_jacobians(const RobustPoseEdge& edge,
                                             kf::Matrix& A,
                                             kf::Matrix& B) const {
    const RobPoseNode& ni = nodes_[edge.from];
    const RobPoseNode& nj = nodes_[edge.to];

    double dx = nj.x - ni.x;
    double dy = nj.y - ni.y;
    double ct = std::cos(ni.theta);
    double st = std::sin(ni.theta);

    A = kf::Matrix(3, 3, 0.0);
    A(0, 0) = -ct;   A(0, 1) = -st;   A(0, 2) = -st * dx + ct * dy;
    A(1, 0) =  st;   A(1, 1) = -ct;   A(1, 2) = -ct * dx - st * dy;
    A(2, 0) =  0.0;  A(2, 1) =  0.0;  A(2, 2) = -1.0;

    B = kf::Matrix(3, 3, 0.0);
    B(0, 0) =  ct;   B(0, 1) = st;    B(0, 2) = 0.0;
    B(1, 0) = -st;   B(1, 1) = ct;    B(1, 2) = 0.0;
    B(2, 0) =  0.0;  B(2, 1) = 0.0;   B(2, 2) = 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// component_log_likelihood
//
// Log-likelihood for one Gaussian component given the current node estimates:
//
//   ll = log_weight + 0.5·log|Ω| - 1.5·log(2π) - 0.5·χ²
//
// where χ² = e^T · Ω · e  (Mahalanobis squared error).
//
// The constant terms (0.5·log|Ω| - 1.5·log(2π)) differ per component and
// must be included so that components with different information matrices are
// compared fairly — a flat null component with tiny |Ω| gets a penalty here.
// ─────────────────────────────────────────────────────────────────────────────

double RobustPoseGraphSlam::component_log_likelihood(
        const RobustPoseEdge& edge, const MMComponent& comp) const {
    double ex, ey, et;
    compute_error(edge, comp, ex, ey, et);

    // Mahalanobis squared error
    kf::Matrix e(3, 1, 0.0);
    e(0, 0) = ex;  e(1, 0) = ey;  e(2, 0) = et;
    kf::Matrix Oe = comp.info * e;
    double chi2 = ex * Oe(0, 0) + ey * Oe(1, 0) + et * Oe(2, 0);

    // Normalisation: 0.5 * log(det(Ω))  —  det is always ≥ 0 for SPD; guard for null
    double d = comp.info.det();
    double log_norm = (d > 1e-300) ? 0.5 * std::log(d) : -1e9;

    static constexpr double k3_log2pi = 1.5 * 1.8378770664093453; // 1.5*log(2π)
    return comp.log_weight + log_norm - k3_log2pi - 0.5 * chi2;
}

// ─────────────────────────────────────────────────────────────────────────────
// select_component – Max-Mixture: pick the component with highest log-likelihood
// ─────────────────────────────────────────────────────────────────────────────

int RobustPoseGraphSlam::select_component(const RobustPoseEdge& edge,
                                           double& best_ll) const {
    int best_k = 0;
    best_ll = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < static_cast<int>(edge.components.size()); ++k) {
        double ll = component_log_likelihood(edge, edge.components[k]);
        if (ll > best_ll) {
            best_ll = ll;
            best_k  = k;
        }
    }
    return best_k;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_weight – robust M-estimator weight  w(χ²)
//
// Huber (industry default):
//   w = 1                 if χ² ≤ δ²   (inlier — full weight)
//   w = δ / √χ²           if χ² > δ²   (outlier — down-weight)
//   Reduces the influence of large errors linearly rather than quadratically.
//
// Cauchy/Lorentzian:
//   w = 1 / (1 + χ²/δ²)
//   Continuously decaying weight; models heavier-tailed distributions.
//   More aggressive outlier rejection but slightly harder to converge.
// ─────────────────────────────────────────────────────────────────────────────

double RobustPoseGraphSlam::kernel_weight(double chi2) const {
    if (chi2 <= 0.0) return 1.0;
    switch (kernel_) {
        case RobustKernel::None:
            return 1.0;
        case RobustKernel::Huber: {
            double delta2 = delta_ * delta_;
            return (chi2 <= delta2) ? 1.0 : delta_ / std::sqrt(chi2);
        }
        case RobustKernel::Cauchy:
            return 1.0 / (1.0 + chi2 / (delta_ * delta_));
    }
    return 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// total_error – Σ  e_k^T · Ω_k · e_k  using each edge's winning component
// ─────────────────────────────────────────────────────────────────────────────

double RobustPoseGraphSlam::total_error() const {
    double err = 0.0;
    for (size_t idx = 0; idx < edges_.size(); ++idx) {
        const auto& edge = edges_[idx];
        const auto& comp = edge.components[sel_[idx]];

        double ex, ey, et;
        compute_error(edge, comp, ex, ey, et);

        kf::Matrix e(3, 1, 0.0);
        e(0, 0) = ex;  e(1, 0) = ey;  e(2, 0) = et;
        kf::Matrix Oe = comp.info * e;
        err += ex * Oe(0, 0) + ey * Oe(1, 0) + et * Oe(2, 0);
    }
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// cholesky_solve – solve H·x = b  (H symmetric positive-definite)
// Identical to PoseGraphSlam::cholesky_solve.
// ─────────────────────────────────────────────────────────────────────────────

kf::Matrix RobustPoseGraphSlam::cholesky_solve(const kf::Matrix& H,
                                                 const kf::Matrix& b) {
    const size_t n = H.rows();
    kf::Matrix L(n, n, 0.0);

    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k)
            sum += L(j, k) * L(j, k);
        double diag = H(j, j) - sum;
        if (diag <= 0.0) diag = 1e-9;
        L(j, j) = std::sqrt(diag);

        for (size_t i = j + 1; i < n; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k)
                s += L(i, k) * L(j, k);
            L(i, j) = (H(i, j) - s) / L(j, j);
        }
    }

    kf::Matrix y(n, 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < i; ++k)
            s += L(i, k) * y(k, 0);
        y(i, 0) = (b(i, 0) - s) / L(i, i);
    }

    kf::Matrix x(n, 1, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = 0.0;
        for (size_t k = static_cast<size_t>(i) + 1; k < n; ++k)
            s += L(k, static_cast<size_t>(i)) * x(k, 0);
        x(static_cast<size_t>(i), 0) =
            (y(static_cast<size_t>(i), 0) - s) /
            L(static_cast<size_t>(i), static_cast<size_t>(i));
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize – Gauss-Newton with Max-Mixture + robust kernel
//
// Per iteration, for each edge:
//   1. Max-Mixture step: select the component k* with highest log-likelihood.
//   2. Compute error e and Mahalanobis χ² = e^T Ω_{k*} e.
//   3. Kernel weight step: w = kernel_weight(χ²).
//   4. Effective information: Ω_eff = w · Ω_{k*}.
//   5. Scatter A^T Ω_eff A, A^T Ω_eff B, etc. into H and b exactly as in
//      PoseGraphSlam — the kernel just scales the information matrix.
//
// The first node is anchored (gauge freedom fix), same as before.
// ─────────────────────────────────────────────────────────────────────────────

int RobustPoseGraphSlam::optimize(
        int max_iterations, double tolerance,
        std::vector<std::vector<RobPoseNode>>* iter_history) {
    const size_t n   = nodes_.size();
    const size_t dim = 3 * n;
    const double anchor = 1e6;

    // Initialise sel_ in case optimize() is called without edges existing yet
    sel_.assign(edges_.size(), 0);

    if (iter_history)
        iter_history->push_back(nodes_);

    int iter = 0;
    for (; iter < max_iterations; ++iter) {
        kf::Matrix H(dim, dim, 0.0);
        kf::Matrix b(dim, 1,   0.0);

        for (size_t idx = 0; idx < edges_.size(); ++idx) {
            const RobustPoseEdge& edge = edges_[idx];

            // ── Step 1: Max-Mixture — select best component ───────────────
            double best_ll;
            int best_k = select_component(edge, best_ll);
            sel_[idx]  = best_k;
            const MMComponent& comp = edge.components[best_k];

            // ── Step 2: Compute error with the winning component ──────────
            double ex, ey, et;
            compute_error(edge, comp, ex, ey, et);

            kf::Matrix e(3, 1, 0.0);
            e(0, 0) = ex;  e(1, 0) = ey;  e(2, 0) = et;

            // ── Step 3: Mahalanobis χ² and kernel weight ──────────────────
            kf::Matrix Oe  = comp.info * e;
            double chi2    = ex * Oe(0, 0) + ey * Oe(1, 0) + et * Oe(2, 0);
            double w       = kernel_weight(chi2);

            // ── Step 4: Weighted information Ω_eff = w · Ω_{k*} ──────────
            kf::Matrix Om_eff = comp.info * w;

            // ── Step 5: Jacobians and block accumulation ──────────────────
            kf::Matrix A(3, 3), B(3, 3);
            compute_jacobians(edge, A, B);

            kf::Matrix At  = A.transpose();
            kf::Matrix Bt  = B.transpose();
            kf::Matrix AtO = At * Om_eff;
            kf::Matrix BtO = Bt * Om_eff;

            kf::Matrix AtOA = AtO * A;
            kf::Matrix AtOB = AtO * B;
            kf::Matrix BtOA = BtO * A;
            kf::Matrix BtOB = BtO * B;
            kf::Matrix AtOe = AtO * e;
            kf::Matrix BtOe = BtO * e;

            size_t ii = 3 * edge.from;
            size_t jj = 3 * edge.to;

            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    H(ii + r, ii + c) += AtOA(r, c);
                    H(ii + r, jj + c) += AtOB(r, c);
                    H(jj + r, ii + c) += BtOA(r, c);
                    H(jj + r, jj + c) += BtOB(r, c);
                }
                b(ii + r, 0) += AtOe(r, 0);
                b(jj + r, 0) += BtOe(r, 0);
            }
        }

        // ── Anchor first node ─────────────────────────────────────────────
        H(0, 0) += anchor;
        H(1, 1) += anchor;
        H(2, 2) += anchor;

        // ── Solve H · Δx = −b ────────────────────────────────────────────
        kf::Matrix neg_b = b * (-1.0);
        kf::Matrix dx    = cholesky_solve(H, neg_b);

        // ── Update poses ──────────────────────────────────────────────────
        double max_update = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double dxi = dx(3*i + 0, 0);
            double dyi = dx(3*i + 1, 0);
            double dti = dx(3*i + 2, 0);
            nodes_[i].x    += dxi;
            nodes_[i].y    += dyi;
            nodes_[i].theta = normalize_angle(nodes_[i].theta + dti);
            max_update = std::max(max_update,
                                  std::sqrt(dxi*dxi + dyi*dyi + dti*dti));
        }

        if (iter_history)
            iter_history->push_back(nodes_);

        if (max_update < tolerance) {
            ++iter;
            break;
        }
    }
    return iter;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

double RobustPoseGraphSlam::normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

} // namespace slam
