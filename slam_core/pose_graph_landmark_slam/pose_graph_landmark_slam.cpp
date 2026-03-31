#include "pose_graph_landmark_slam.h"

#include <cmath>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

PoseGraphLandmarkSlam::PoseGraphLandmarkSlam() = default;

// ─────────────────────────────────────────────────────────────────────────────
// Graph building
// ─────────────────────────────────────────────────────────────────────────────

size_t PoseGraphLandmarkSlam::add_pose(double x, double y, double theta) {
    poses_.push_back({x, y, theta});
    return poses_.size() - 1;
}

size_t PoseGraphLandmarkSlam::add_landmark(double x, double y) {
    landmarks_.push_back({x, y});
    return landmarks_.size() - 1;
}

size_t PoseGraphLandmarkSlam::add_odometry_edge(size_t from, size_t to,
                                                 double dx, double dy,
                                                 double dtheta,
                                                 double info_pos,
                                                 double info_rot) {
    kf::Matrix info(3, 3, 0.0);
    info(0, 0) = info_pos;
    info(1, 1) = info_pos;
    info(2, 2) = info_rot;
    odom_edges_.push_back({from, to, dx, dy, dtheta, info});
    return odom_edges_.size() - 1;
}

size_t PoseGraphLandmarkSlam::add_landmark_edge(size_t pose_idx,
                                                 size_t landmark_idx,
                                                 double dx, double dy,
                                                 double info_pos) {
    kf::Matrix info(2, 2, 0.0);
    info(0, 0) = info_pos;
    info(1, 1) = info_pos;
    lm_edges_.push_back({pose_idx, landmark_idx, dx, dy, info});
    return lm_edges_.size() - 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_odom_error – same as PoseGraphSlam::compute_error
//
// Predicted relative pose from i to j (in frame i):
//     dx_pred = cos(θ_i)*(x_j − x_i) + sin(θ_i)*(y_j − y_i)
//     dy_pred = −sin(θ_i)*(x_j − x_i) + cos(θ_i)*(y_j − y_i)
//     dθ_pred = θ_j − θ_i
// Error = predicted − measured
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphLandmarkSlam::compute_odom_error(const PLOdometryEdge& e,
                                                double& ex, double& ey,
                                                double& et) const {
    const PLPoseNode& ni = poses_[e.from];
    const PLPoseNode& nj = poses_[e.to];

    double dx = nj.x - ni.x;
    double dy = nj.y - ni.y;
    double ct = std::cos(ni.theta);
    double st = std::sin(ni.theta);

    double dx_pred =  ct * dx + st * dy;
    double dy_pred = -st * dx + ct * dy;
    double dt_pred = nj.theta - ni.theta;

    ex = dx_pred - e.dx;
    ey = dy_pred - e.dy;
    et = normalize_angle(dt_pred - e.dtheta);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_odom_jacobians – same as PoseGraphSlam::compute_jacobians
//
// A = ∂e/∂[x_i, y_i, θ_i]  (3×3)
// B = ∂e/∂[x_j, y_j, θ_j]  (3×3)
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphLandmarkSlam::compute_odom_jacobians(const PLOdometryEdge& e,
                                                    kf::Matrix& A,
                                                    kf::Matrix& B) const {
    const PLPoseNode& ni = poses_[e.from];
    const PLPoseNode& nj = poses_[e.to];

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
// compute_lm_error – mismatch between predicted and measured landmark observation
//
// Given robot pose i at (p_x, p_y, θ) and landmark l at (l_x, l_y):
// The predicted observation in the robot frame is:
//     dlx    = l_x − p_x
//     dly    = l_y − p_y
//     z_pred = R_i^T · [dlx, dly]^T
//            = [cos(θ)·dlx + sin(θ)·dly,  -sin(θ)·dlx + cos(θ)·dly]
//
// Error:  e = z_pred − z_meas   (2×1)
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphLandmarkSlam::compute_lm_error(const PLLandmarkEdge& e,
                                              double& ex, double& ey) const {
    const PLPoseNode&     pi = poses_[e.pose_idx];
    const PLLandmarkNode& lm = landmarks_[e.landmark_idx];

    double dlx = lm.x - pi.x;
    double dly = lm.y - pi.y;
    double ct  = std::cos(pi.theta);
    double st  = std::sin(pi.theta);

    double zx =  ct * dlx + st * dly;
    double zy = -st * dlx + ct * dly;

    ex = zx - e.dx;
    ey = zy - e.dy;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_lm_jacobians
//
// A = ∂e/∂[x_i, y_i, θ_i]   (2×3)   w.r.t. the observing robot pose
// B = ∂e/∂[lx_l, ly_l]      (2×2)   w.r.t. the landmark
//
// Derivation (let dlx = lm.x − p_x,  dly = lm.y − p_y):
//   ∂z_pred_x/∂p_x  = −cos(θ)
//   ∂z_pred_x/∂p_y  = −sin(θ)
//   ∂z_pred_x/∂θ    = −sin(θ)·dlx + cos(θ)·dly
//
//   ∂z_pred_y/∂p_x  =  sin(θ)
//   ∂z_pred_y/∂p_y  = −cos(θ)
//   ∂z_pred_y/∂θ    = −cos(θ)·dlx − sin(θ)·dly
//
//   ∂z_pred_x/∂lm_x =  cos(θ),   ∂z_pred_x/∂lm_y =  sin(θ)
//   ∂z_pred_y/∂lm_x = −sin(θ),   ∂z_pred_y/∂lm_y =  cos(θ)
//
// B = R_i^T  (the robot's rotation matrix transposed)
// ─────────────────────────────────────────────────────────────────────────────

void PoseGraphLandmarkSlam::compute_lm_jacobians(const PLLandmarkEdge& e,
                                                  kf::Matrix& A,
                                                  kf::Matrix& B) const {
    const PLPoseNode&     pi = poses_[e.pose_idx];
    const PLLandmarkNode& lm = landmarks_[e.landmark_idx];

    double dlx = lm.x - pi.x;
    double dly = lm.y - pi.y;
    double ct  = std::cos(pi.theta);
    double st  = std::sin(pi.theta);

    A = kf::Matrix(2, 3, 0.0);
    A(0, 0) = -ct;   A(0, 1) = -st;   A(0, 2) = -st * dlx + ct * dly;
    A(1, 0) =  st;   A(1, 1) = -ct;   A(1, 2) = -ct * dlx - st * dly;

    B = kf::Matrix(2, 2, 0.0);
    B(0, 0) =  ct;   B(0, 1) =  st;
    B(1, 0) = -st;   B(1, 1) =  ct;
}

// ─────────────────────────────────────────────────────────────────────────────
// cholesky_solve – solve H·x = b  (H symmetric positive-definite)
// Identical to PoseGraphSlam::cholesky_solve.
// ─────────────────────────────────────────────────────────────────────────────

kf::Matrix PoseGraphLandmarkSlam::cholesky_solve(const kf::Matrix& H,
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
// total_error – Σ e^T Ω e  over all edges
// ─────────────────────────────────────────────────────────────────────────────

double PoseGraphLandmarkSlam::total_error() const {
    double err = 0.0;

    for (const auto& e : odom_edges_) {
        double ex, ey, et;
        compute_odom_error(e, ex, ey, et);
        kf::Matrix ev(3, 1, 0.0);
        ev(0, 0) = ex;  ev(1, 0) = ey;  ev(2, 0) = et;
        kf::Matrix oe = e.info * ev;
        err += ev(0, 0)*oe(0, 0) + ev(1, 0)*oe(1, 0) + ev(2, 0)*oe(2, 0);
    }

    for (const auto& e : lm_edges_) {
        double ex, ey;
        compute_lm_error(e, ex, ey);
        kf::Matrix ev(2, 1, 0.0);
        ev(0, 0) = ex;  ev(1, 0) = ey;
        kf::Matrix oe = e.info * ev;
        err += ev(0, 0)*oe(0, 0) + ev(1, 0)*oe(1, 0);
    }
    return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize – Gauss-Newton with LM-style damping
//
// State vector: [poses (3-DOF each) | landmarks (2-DOF each)]
// Linear system per iteration:  (H + λI) · Δx = −b
//
// λ (damping) keeps the system positive-definite even when landmark-only
// constraints leave H rank-deficient.
// The first robot pose is also anchored with a large diagonal to fix gauge
// freedom (the same as PoseGraphSlam).
// ─────────────────────────────────────────────────────────────────────────────

int PoseGraphLandmarkSlam::optimize(int max_iterations, double tolerance,
                                     double damping,
                                     std::vector<PLSnapshot>* iter_history) {
    const size_t np = poses_.size();
    const size_t nl = landmarks_.size();
    const size_t dim = system_dim();
    const double anchor = 1e6;

    if (iter_history) {
        iter_history->push_back({poses_, landmarks_});
    }

    int iter = 0;
    for (; iter < max_iterations; ++iter) {
        kf::Matrix H(dim, dim, 0.0);
        kf::Matrix b(dim, 1,   0.0);

        // ── Odometry edges (pose-to-pose, 3-DOF ↔ 3-DOF) ────────────────────
        for (const auto& e : odom_edges_) {
            double ex, ey, et;
            compute_odom_error(e, ex, ey, et);

            kf::Matrix A(3, 3), B(3, 3);
            compute_odom_jacobians(e, A, B);

            kf::Matrix ev(3, 1, 0.0);
            ev(0, 0) = ex;  ev(1, 0) = ey;  ev(2, 0) = et;

            const kf::Matrix& Om = e.info;
            kf::Matrix AtO  = A.transpose() * Om;
            kf::Matrix BtO  = B.transpose() * Om;
            kf::Matrix AtOA = AtO * A;
            kf::Matrix AtOB = AtO * B;
            kf::Matrix BtOA = BtO * A;
            kf::Matrix BtOB = BtO * B;
            kf::Matrix AtOe = AtO * ev;
            kf::Matrix BtOe = BtO * ev;

            size_t ii = pose_offset(e.from);
            size_t jj = pose_offset(e.to);

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

        // ── Landmark observation edges (pose 3-DOF ↔ landmark 2-DOF) ─────────
        for (const auto& e : lm_edges_) {
            double ex, ey;
            compute_lm_error(e, ex, ey);

            kf::Matrix A(2, 3), B(2, 2);
            compute_lm_jacobians(e, A, B);

            kf::Matrix ev(2, 1, 0.0);
            ev(0, 0) = ex;  ev(1, 0) = ey;

            const kf::Matrix& Om = e.info;     // 2×2
            kf::Matrix AtO  = A.transpose() * Om;   // 3×2
            kf::Matrix BtO  = B.transpose() * Om;   // 2×2
            kf::Matrix AtOA = AtO * A;               // 3×3
            kf::Matrix AtOB = AtO * B;               // 3×2
            kf::Matrix BtOA = BtO * A;               // 2×3
            kf::Matrix BtOB = BtO * B;               // 2×2
            kf::Matrix AtOe = AtO * ev;              // 3×1
            kf::Matrix BtOe = BtO * ev;              // 2×1

            size_t ii = pose_offset(e.pose_idx);          // 3-DOF block
            size_t jj = lm_offset(e.landmark_idx);        // 2-DOF block

            // 3×3 pose-pose block
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c)
                    H(ii + r, ii + c) += AtOA(r, c);
                b(ii + r, 0) += AtOe(r, 0);
            }
            // 2×2 landmark-landmark block
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 2; ++c)
                    H(jj + r, jj + c) += BtOB(r, c);
                b(jj + r, 0) += BtOe(r, 0);
            }
            // 3×2 pose-landmark cross blocks
            for (size_t r = 0; r < 3; ++r)
                for (size_t c = 0; c < 2; ++c) {
                    H(ii + r, jj + c) += AtOB(r, c);
                    H(jj + c, ii + r) += AtOB(r, c);  // symmetric
                }
        }

        // ── Damping: (H + λI) – stabilises rank-deficient landmark systems ───
        for (size_t d = 0; d < dim; ++d)
            H(d, d) += damping;

        // ── Anchor first pose to fix gauge freedom ───────────────────────────
        H(0, 0) += anchor;
        H(1, 1) += anchor;
        H(2, 2) += anchor;

        // ── Solve (H + λI) · Δx = −b ─────────────────────────────────────────
        kf::Matrix neg_b = b * (-1.0);
        kf::Matrix dx    = cholesky_solve(H, neg_b);

        // ── Update poses ──────────────────────────────────────────────────────
        double max_update = 0.0;
        for (size_t i = 0; i < np; ++i) {
            double dxi = dx(pose_offset(i) + 0, 0);
            double dyi = dx(pose_offset(i) + 1, 0);
            double dti = dx(pose_offset(i) + 2, 0);
            poses_[i].x     += dxi;
            poses_[i].y     += dyi;
            poses_[i].theta  = normalize_angle(poses_[i].theta + dti);
            max_update = std::max(max_update,
                                  std::sqrt(dxi*dxi + dyi*dyi + dti*dti));
        }

        // ── Update landmarks ──────────────────────────────────────────────────
        for (size_t l = 0; l < nl; ++l) {
            double dlx = dx(lm_offset(l) + 0, 0);
            double dly = dx(lm_offset(l) + 1, 0);
            landmarks_[l].x += dlx;
            landmarks_[l].y += dly;
            max_update = std::max(max_update,
                                  std::sqrt(dlx*dlx + dly*dly));
        }

        if (iter_history) {
            iter_history->push_back({poses_, landmarks_});
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

double PoseGraphLandmarkSlam::normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

} // namespace slam
