#include "pose_graph_slam.h"

#include <cmath>
#include <limits>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

PoseGraphSlam::PoseGraphSlam() = default;

// ─────────────────────────────────────────────────────────────────────────────
// Graph building
// ─────────────────────────────────────────────────────────────────────────────

size_t PoseGraphSlam::add_node(double x, double y, double theta) {
    nodes_.push_back({x, y, theta});
    return nodes_.size() - 1;
}

size_t PoseGraphSlam::add_edge(size_t from, size_t to,
                               double dx, double dy, double dtheta,
                               double info_pos, double info_rot) {
    kf::Matrix info(3, 3, 0.0);
    info(0, 0) = info_pos;
    info(1, 1) = info_pos;
    info(2, 2) = info_rot;
    edges_.push_back({from, to, dx, dy, dtheta, info});
    return edges_.size() - 1;
}

void PoseGraphSlam::update_edge_info(size_t edge_idx,
                                      double info_pos, double info_rot) {
    kf::Matrix info(3, 3, 0.0);
    info(0, 0) = info_pos;
    info(1, 1) = info_pos;
    info(2, 2) = info_rot;
    edges_[edge_idx].info = info;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_error – the mismatch between predicted and measured relative pose
//
// Predicted relative pose from i to j (in frame i):
//     dx_pred = cos(θ_i)*(x_j − x_i) + sin(θ_i)*(y_j − y_i)
//     dy_pred = −sin(θ_i)*(x_j − x_i) + cos(θ_i)*(y_j − y_i)
//     dθ_pred = θ_j − θ_i
//
// Error = predicted − measured
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphSlam::compute_error(const PoseEdge& edge,
                                   double& ex, double& ey, double& et) const {
    const PoseNode& ni = nodes_[edge.from];
    const PoseNode& nj = nodes_[edge.to];

    double dx = nj.x - ni.x;
    double dy = nj.y - ni.y;
    double ct = std::cos(ni.theta);
    double st = std::sin(ni.theta);

    double dx_pred =  ct * dx + st * dy;
    double dy_pred = -st * dx + ct * dy;
    double dt_pred = nj.theta - ni.theta;

    ex = dx_pred - edge.dx;
    ey = dy_pred - edge.dy;
    et = normalize_angle(dt_pred - edge.dtheta);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_jacobians – Jacobians of the error w.r.t. nodes i and j
//
// A = ∂e/∂[x_i, y_i, θ_i]   (3×3)
// B = ∂e/∂[x_j, y_j, θ_j]   (3×3)
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphSlam::compute_jacobians(const PoseEdge& edge,
                                       kf::Matrix& A, kf::Matrix& B) const {
    const PoseNode& ni = nodes_[edge.from];
    const PoseNode& nj = nodes_[edge.to];

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
// cholesky_solve – solve H·x = b  where H is symmetric positive-definite
//
// Uses Cholesky decomposition H = L·L^T, then:
//   forward-substitution  L·y = b
//   backward-substitution L^T·x = y
// ─────────────────────────────────────────────────────────────────────────────

kf::Matrix PoseGraphSlam::cholesky_solve(const kf::Matrix& H,
                                          const kf::Matrix& b) {
    const size_t n = H.rows();
    kf::Matrix L(n, n, 0.0);

    // Cholesky decomposition
    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k)
            sum += L(j, k) * L(j, k);
        double diag = H(j, j) - sum;
        if (diag <= 0.0) diag = 1e-9;          // numerical guard
        L(j, j) = std::sqrt(diag);

        for (size_t i = j + 1; i < n; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k)
                s += L(i, k) * L(j, k);
            L(i, j) = (H(i, j) - s) / L(j, j);
        }
    }

    // Forward substitution: L·y = b
    kf::Matrix y(n, 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < i; ++k)
            s += L(i, k) * y(k, 0);
        y(i, 0) = (b(i, 0) - s) / L(i, i);
    }

    // Backward substitution: L^T·x = y
    kf::Matrix x(n, 1, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = 0.0;
        for (size_t k = static_cast<size_t>(i) + 1; k < n; ++k)
            s += L(k, static_cast<size_t>(i)) * x(k, 0);
        x(static_cast<size_t>(i), 0) =
            (y(static_cast<size_t>(i), 0) - s) / L(static_cast<size_t>(i),
                                                     static_cast<size_t>(i));
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// assemble_hessian – build H at the current linearisation point
//
// Same accumulation loop as optimize() but WITHOUT anchor augmentation and
// WITHOUT building the gradient b or solving.  Used by hierarchical SLAM to
// compute the Schur complement of interior fine nodes.
// ─────────────────────────────────────────────────────────────────────────────

kf::Matrix PoseGraphSlam::assemble_hessian() const {
    const size_t dim = 3 * nodes_.size();
    kf::Matrix H(dim, dim, 0.0);

    for (const auto& edge : edges_) {
        kf::Matrix A(3, 3), B(3, 3);
        compute_jacobians(edge, A, B);

        const kf::Matrix& Om = edge.info;
        kf::Matrix AtO  = A.transpose() * Om;
        kf::Matrix BtO  = B.transpose() * Om;
        kf::Matrix AtOA = AtO * A;
        kf::Matrix AtOB = AtO * B;
        kf::Matrix BtOA = BtO * A;
        kf::Matrix BtOB = BtO * B;

        size_t ii = 3 * edge.from;
        size_t jj = 3 * edge.to;

        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                H(ii + r, ii + c) += AtOA(r, c);
                H(ii + r, jj + c) += AtOB(r, c);
                H(jj + r, ii + c) += BtOA(r, c);
                H(jj + r, jj + c) += BtOB(r, c);
            }
        }
    }
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// marginal_covariances – diagonal blocks of (H + anchor)^{-1}
//
// Returns one 3×3 covariance matrix per node.  The anchor (1e6 diagonal on
// node 0) is kept so the system stays non-singular; the resulting covariance
// for node 0 will be near-zero (strongly pinned).
// ─────────────────────────────────────────────────────────────────────────────

std::vector<kf::Matrix> PoseGraphSlam::marginal_covariances() const {
    if (nodes_.empty()) return {};

    const size_t n   = nodes_.size();
    const size_t dim = 3 * n;
    const double anchor = 1e6;

    // Build H with anchor (same as optimize()).
    kf::Matrix H(dim, dim, 0.0);
    for (const auto& edge : edges_) {
        kf::Matrix A(3, 3), B(3, 3);
        compute_jacobians(edge, A, B);

        const kf::Matrix& Om = edge.info;
        kf::Matrix AtO  = A.transpose() * Om;
        kf::Matrix BtO  = B.transpose() * Om;

        size_t ii = 3 * edge.from;
        size_t jj = 3 * edge.to;

        kf::Matrix AtOA = AtO * A;
        kf::Matrix AtOB = AtO * B;
        kf::Matrix BtOA = BtO * A;
        kf::Matrix BtOB = BtO * B;

        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                H(ii + r, ii + c) += AtOA(r, c);
                H(ii + r, jj + c) += AtOB(r, c);
                H(jj + r, ii + c) += BtOA(r, c);
                H(jj + r, jj + c) += BtOB(r, c);
            }
        }
    }
    H(0, 0) += anchor;
    H(1, 1) += anchor;
    H(2, 2) += anchor;

    kf::Matrix Sigma = H.inverse();     // (dim×dim) full covariance

    std::vector<kf::Matrix> covs;
    covs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        kf::Matrix cov(3, 3, 0.0);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c)
                cov(r, c) = Sigma(3 * i + r, 3 * i + c);
        covs.push_back(cov);
    }
    return covs;
}

// ─────────────────────────────────────────────────────────────────────────────
// total_error – sum of  e_ij^T · Ω_ij · e_ij  over all edges
// ─────────────────────────────────────────────────────────────────────────────

double PoseGraphSlam::total_error() const {
    double err = 0.0;
    for (const auto& edge : edges_) {
        double ex, ey, et;
        compute_error(edge, ex, ey, et);

        kf::Matrix e(3, 1, 0.0);
        e(0, 0) = ex;  e(1, 0) = ey;  e(2, 0) = et;

        kf::Matrix oe = edge.info * e;          // Ω · e
        err += e(0, 0) * oe(0, 0) + e(1, 0) * oe(1, 0) + e(2, 0) * oe(2, 0);
    }
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize – Gauss-Newton iteration
//
// Build the linear system  H·Δx = −b  from all edges, solve, update poses.
// The first node is anchored (large diagonal addition) to fix gauge freedom.
// ─────────────────────────────────────────────────────────────────────────────

int PoseGraphSlam::optimize(int max_iterations, double tolerance,
                             std::vector<std::vector<PoseNode>>* iter_history) {
    const size_t n = nodes_.size();
    const size_t dim = 3 * n;
    const double anchor = 1e6;          // large value to fix first node

    if (iter_history) {
        iter_history->push_back(nodes_);    // snapshot before iteration 0
    }

    int iter = 0;
    for (; iter < max_iterations; ++iter) {
        kf::Matrix H(dim, dim, 0.0);
        kf::Matrix b(dim, 1, 0.0);

        // Accumulate contributions from every edge
        for (const auto& edge : edges_) {
            double ex, ey, et;
            compute_error(edge, ex, ey, et);

            kf::Matrix A(3, 3), B(3, 3);
            compute_jacobians(edge, A, B);

            kf::Matrix e(3, 1, 0.0);
            e(0, 0) = ex;  e(1, 0) = ey;  e(2, 0) = et;

            const kf::Matrix& Om = edge.info;
            kf::Matrix At = A.transpose();
            kf::Matrix Bt = B.transpose();

            // 3×3 block products
            kf::Matrix AtO  = At * Om;
            kf::Matrix BtO  = Bt * Om;
            kf::Matrix AtOA = AtO * A;
            kf::Matrix AtOB = AtO * B;
            kf::Matrix BtOA = BtO * A;
            kf::Matrix BtOB = BtO * B;
            kf::Matrix AtOe = AtO * e;
            kf::Matrix BtOe = BtO * e;

            size_t ii = 3 * edge.from;
            size_t jj = 3 * edge.to;

            // Scatter 3×3 blocks into H and b
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

        // Anchor first node to fix gauge freedom
        H(0, 0) += anchor;
        H(1, 1) += anchor;
        H(2, 2) += anchor;

        // Solve H · dx = −b
        kf::Matrix neg_b = b * (-1.0);
        kf::Matrix dx = cholesky_solve(H, neg_b);

        // Update poses
        double max_update = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double dxi = dx(3 * i + 0, 0);
            double dyi = dx(3 * i + 1, 0);
            double dti = dx(3 * i + 2, 0);
            nodes_[i].x     += dxi;
            nodes_[i].y     += dyi;
            nodes_[i].theta  = normalize_angle(nodes_[i].theta + dti);

            max_update = std::max(max_update,
                                  std::sqrt(dxi * dxi + dyi * dyi + dti * dti));
        }

        if (iter_history) {
            iter_history->push_back(nodes_);
        }

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

double PoseGraphSlam::normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

} // namespace slam
