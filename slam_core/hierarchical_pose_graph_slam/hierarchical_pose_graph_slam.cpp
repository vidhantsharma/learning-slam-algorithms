#include "hierarchical_pose_graph_slam.h"

#include <cmath>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

HierarchicalPoseGraphSlam::HierarchicalPoseGraphSlam(
    size_t keyframe_interval, double fine_info_pos, double fine_info_rot)
    : keyframe_interval_(keyframe_interval)
    , fine_info_pos_(fine_info_pos)
    , fine_info_rot_(fine_info_rot)
    // kf_info is a linear-chain approximation of the Schur complement:
    // marginalising K internal fine nodes yields effective info ≈ fine_info / K.
    // The exact value requires computing the Schur complement of the local H
    // matrix (H_bb − H_bi H_ii^{-1} H_ib), which would need access to the
    // assembled H from each window's fine-level graph.
    , kf_info_pos_(fine_info_pos  / static_cast<double>(keyframe_interval))
    , kf_info_rot_(fine_info_rot  / static_cast<double>(keyframe_interval))
{
    // Seed both levels with the origin pose (known start).
    fine_nodes_.push_back({0.0, 0.0, 0.0});
    fine_covariances_.push_back(kf::Matrix(3, 3, 0.0));   // origin: no uncertainty
    keyframe_fine_indices_.push_back(0);
    global_graph_.add_node(0.0, 0.0, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// add_pose – online: one odometry step at a time
//
// Three things happen every call:
//   1. Propagate the new FineNode from the previous pose + raw odometry.
//   2. Compose the odometry into the pre-integration accumulator (SE(2) chain).
//   3. If the accumulator has K steps, promote a new keyframe into the global
//      graph with the accumulated relative pose as the edge measurement.
// ─────────────────────────────────────────────────────────────────────────────

size_t HierarchicalPoseGraphSlam::add_pose(double dx, double dy, double dtheta)
{
    // ── 1. Compute and store the new fine node ────────────────────────────
    const FineNode& prev = fine_nodes_.back();
    double ct = std::cos(prev.theta);
    double st = std::sin(prev.theta);

    FineNode next;
    next.x     = prev.x + ct * dx - st * dy;
    next.y     = prev.y + st * dx + ct * dy;
    next.theta = normalize_angle(prev.theta + dtheta);

    size_t new_idx = fine_nodes_.size();
    fine_nodes_.push_back(next);
    fine_covariances_.push_back(kf::Matrix(3, 3, 0.0));   // filled by propagate_corrections
    fine_edges_.push_back({new_idx - 1, new_idx, dx, dy, dtheta});

    // ── 2. SE(2) pre-integration ──────────────────────────────────────────
    // Compose (dx, dy, dtheta) — given in the current fine pose's frame —
    // into accum_*, which accumulates the relative pose from the last keyframe.
    //
    // Because each raw measurement is expressed in the LOCAL frame of the
    // fine pose that produced it, and the accumulated pose has already
    // rotated by accum_theta_, we rotate dx/dy by accum_theta_ before adding.
    double ca = std::cos(accum_theta_);
    double sa = std::sin(accum_theta_);
    accum_x_    += ca * dx - sa * dy;
    accum_y_    += sa * dx + ca * dy;
    accum_theta_ = normalize_angle(accum_theta_ + dtheta);
    ++steps_in_window_;

    // ── 3. Promote keyframe when window is complete ───────────────────────
    if (steps_in_window_ == keyframe_interval_) {
        size_t prev_kf = global_graph_.nodes().size() - 1;
        size_t new_kf  = global_graph_.add_node(next.x, next.y, next.theta);

        // The pre-integrated (accum_x_, accum_y_, accum_theta_) is the
        // measured relative pose of new_kf in the LOCAL frame of prev_kf.
        // Initial info uses the fine_info/K approximation; optimize_local_window
        // will replace this with the exact Schur complement once the window is
        // processed after global optimisation.
        size_t eidx = global_graph_.add_edge(prev_kf, new_kf,
                               accum_x_, accum_y_, accum_theta_,
                               kf_info_pos_, kf_info_rot_);
        kf_odom_edge_indices_.push_back(eidx);

        keyframe_fine_indices_.push_back(new_idx);

        // Reset accumulator for the next window.
        accum_x_ = accum_y_ = accum_theta_ = 0.0;
        steps_in_window_ = 0;
    }

    return new_idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// add_loop_closure
// ─────────────────────────────────────────────────────────────────────────────

void HierarchicalPoseGraphSlam::add_loop_closure(
    size_t from_kf, size_t to_kf,
    double dx, double dy, double dtheta,
    double info_pos, double info_rot)
{
    global_graph_.add_edge(from_kf, to_kf, dx, dy, dtheta, info_pos, info_rot);
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize
// ─────────────────────────────────────────────────────────────────────────────

int HierarchicalPoseGraphSlam::optimize(
    int max_iterations, double tolerance,
    std::vector<std::vector<PoseNode>>* kf_iter_history)
{
    int iters = global_graph_.optimize(max_iterations, tolerance, kf_iter_history);
    propagate_corrections();
    return iters;
}

// ─────────────────────────────────────────────────────────────────────────────
// propagate_corrections
//
// For each keyframe window [KF_k … KF_{k+1}):
//   a) Snap fine_nodes_[KF_k] to the globally corrected keyframe pose.
//   b) Re-roll the stored fine odometry forward within the window.
//
// Because the outer loop processes windows in order (kf = 0, 1, 2, …),
// fine_nodes_[KF_{k+1}] is first written by the re-roll of window k,
// then immediately overwritten when window k+1 sets its own anchor.
// This is correct: the global graph provides the authoritative position
// for every keyframe node.
// ─────────────────────────────────────────────────────────────────────────────

void HierarchicalPoseGraphSlam::propagate_corrections()
{
    const size_t num_kf = keyframe_fine_indices_.size();

    for (size_t kf = 0; kf < num_kf; ++kf) {
        const bool has_next_kf = (kf + 1 < num_kf);

        if (has_next_kf) {
            // Complete window: run local Gauss-Newton with both KF endpoints
            // pinned.  This smoothly distributes the global correction across
            // all interior fine nodes rather than just re-rolling from one end.
            optimize_local_window(kf);
        } else {
            // Trailing incomplete window (between last KF and current robot
            // position): no second KF anchor yet, so fall back to
            // dead-reckoning from the corrected last keyframe.
            const auto& kf_nodes  = global_graph_.nodes();
            size_t fine_start     = keyframe_fine_indices_[kf];
            size_t fine_end       = fine_nodes_.size() - 1;

            fine_nodes_[fine_start].x     = kf_nodes[kf].x;
            fine_nodes_[fine_start].y     = kf_nodes[kf].y;
            fine_nodes_[fine_start].theta = kf_nodes[kf].theta;

            for (size_t fi = fine_start; fi < fine_end; ++fi) {
                const FineEdge& e = fine_edges_[fi];
                double ct = std::cos(fine_nodes_[fi].theta);
                double st = std::sin(fine_nodes_[fi].theta);
                fine_nodes_[fi + 1].x     = fine_nodes_[fi].x + ct * e.dx - st * e.dy;
                fine_nodes_[fi + 1].y     = fine_nodes_[fi].y + st * e.dx + ct * e.dy;
                fine_nodes_[fi + 1].theta = normalize_angle(fine_nodes_[fi].theta + e.dtheta);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize_local_window
//
// Builds a small PoseGraphSlam for the K+1 nodes in window [KF_k, KF_{k+1}]:
//   node 0        = KF_k   (corrected position from global graph)
//   node 1..K-1   = interior fine nodes
//   node K        = KF_{k+1} (corrected position from global graph)
//
// Edges:
//   K fine odometry edges between consecutive local nodes (fine_info weights).
//   One strong virtual edge from node 0 to node K encoding the globally
//   corrected KF-to-KF relative pose.  This effectively pins node K in place,
//   so both endpoints are fixed and the interior nodes are optimised freely.
//
// The approach replaces dead-reckoning within the window and distributes the
// global correction smoothly across all interior fine nodes.
// ─────────────────────────────────────────────────────────────────────────────

void HierarchicalPoseGraphSlam::optimize_local_window(size_t kf)
{
    const auto& kf_corrected = global_graph_.nodes();
    size_t fine_start = keyframe_fine_indices_[kf];
    size_t fine_end   = keyframe_fine_indices_[kf + 1];
    size_t n_local    = fine_end - fine_start + 1;   // K + 1 nodes

    // ── Build local graph ─────────────────────────────────────────────────
    PoseGraphSlam local;

    // Node 0: corrected KF_k (will be anchored automatically by PoseGraphSlam).
    local.add_node(kf_corrected[kf].x,
                   kf_corrected[kf].y,
                   kf_corrected[kf].theta);

    // Nodes 1..K-1: interior fine nodes at their current dead-reckoned positions.
    for (size_t fi = fine_start + 1; fi < fine_end; ++fi)
        local.add_node(fine_nodes_[fi].x, fine_nodes_[fi].y, fine_nodes_[fi].theta);

    // Node K: corrected KF_{k+1} (initial guess; will be pinned via virtual edge).
    local.add_node(kf_corrected[kf + 1].x,
                   kf_corrected[kf + 1].y,
                   kf_corrected[kf + 1].theta);

    // ── Fine odometry edges ───────────────────────────────────────────────
    for (size_t fi = fine_start; fi < fine_end; ++fi) {
        const FineEdge& e  = fine_edges_[fi];
        size_t local_from  = fi - fine_start;
        local.add_edge(local_from, local_from + 1,
                       e.dx, e.dy, e.dtheta,
                       fine_info_pos_, fine_info_rot_);
    }

    // ── Virtual anchor edge: node 0 → node K with corrected relative pose ─
    // Pinning node K stops it from drifting despite only node 0 being formally
    // anchored by PoseGraphSlam.  A very high information weight is used so
    // node K deviates negligibly from the globally corrected KF_{k+1} position.
    double dx_w  = kf_corrected[kf + 1].x - kf_corrected[kf].x;
    double dy_w  = kf_corrected[kf + 1].y - kf_corrected[kf].y;
    double dth   = normalize_angle(kf_corrected[kf + 1].theta - kf_corrected[kf].theta);
    double ct    = std::cos(kf_corrected[kf].theta);
    double st    = std::sin(kf_corrected[kf].theta);
    double dx_kf =  ct * dx_w + st * dy_w;
    double dy_kf = -st * dx_w + ct * dy_w;

    const double anchor_info = 1e6;   // effectively pins node K
    local.add_edge(0, n_local - 1, dx_kf, dy_kf, dth, anchor_info, anchor_info);

    // ── Optimise local window ─────────────────────────────────────────────
    local.optimize(20, 1e-6);

    // ── Exact Schur complement for the KF odometry edge info ─────────────
    // Build a measurement-only local graph (no virtual anchor edge) at the
    // optimised node positions, then compute the Schur complement of the
    // interior nodes to get the effective 6×6 information matrix for the
    // two boundary keyframe DOFs.
    //
    // For a linear chain with isotropic noise, this equals fine_info/K
    // exactly.  For windows that span corners (non-zero curvature, whose
    // Jacobians A,B deviate from −I and I), the Schur complement differs
    // from the heuristic, yielding a more accurate kf edge weight.
    if (n_local > 2) {
        const auto& ln = local.nodes();

        // Measurement-only graph: same topology, no virtual anchor edge.
        PoseGraphSlam local_meas;
        for (size_t li = 0; li < n_local; ++li)
            local_meas.add_node(ln[li].x, ln[li].y, ln[li].theta);
        for (size_t fi = fine_start; fi < fine_end; ++fi) {
            const FineEdge& e = fine_edges_[fi];
            size_t lf = fi - fine_start;
            local_meas.add_edge(lf, lf + 1, e.dx, e.dy, e.dtheta,
                                fine_info_pos_, fine_info_rot_);
        }

        kf::Matrix H_meas = local_meas.assemble_hessian();

        // Partition H_meas into boundary (nodes 0 and K) vs interior blocks.
        size_t K_idx = n_local - 1;       // local index of KF_{k+1}
        size_t n_i   = 3 * (K_idx - 1);  // interior DOFs (nodes 1..K-1)

        // H_bb (6×6): self and cross info for boundary nodes.
        kf::Matrix H_bb(6, 6, 0.0);
        for (size_t r = 0; r < 3; ++r) for (size_t c = 0; c < 3; ++c) {
            H_bb(r,   c)   = H_meas(r,             c);
            H_bb(r,   3+c) = H_meas(r,             3*K_idx+c);
            H_bb(3+r, c)   = H_meas(3*K_idx+r,    c);
            H_bb(3+r, 3+c) = H_meas(3*K_idx+r,    3*K_idx+c);
        }

        // H_ii (n_i × n_i): interior block, rows/cols 3..3K-1.
        kf::Matrix H_ii(n_i, n_i, 0.0);
        for (size_t r = 0; r < n_i; ++r)
            for (size_t c = 0; c < n_i; ++c)
                H_ii(r, c) = H_meas(3+r, 3+c);

        // H_bi (6 × n_i): boundary rows, interior cols.
        kf::Matrix H_bi(6, n_i, 0.0);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < n_i; ++c) {
                H_bi(r,   c) = H_meas(r,          3+c);
                H_bi(3+r, c) = H_meas(3*K_idx+r,  3+c);
            }

        // Schur complement: S = H_bb − H_bi H_ii^{−1} H_bi^T (6×6).
        // S[0:3,0:3] = effective info of KF_k  from this window.
        // S[3:6,3:6] = effective info of KF_{k+1} from this window.
        kf::Matrix S = H_bb - H_bi * H_ii.inverse() * H_bi.transpose();

        // Conservative scalar extraction: minimum of info at both endpoints.
        // For a straight chain S_00 == S_KK; for curved windows they diverge.
        double ip_0 = 0.5 * (S(0, 0) + S(1, 1));
        double ip_K = 0.5 * (S(3, 3) + S(4, 4));
        double ir_0 = S(2, 2);
        double ir_K = S(5, 5);
        double info_pos_exact = std::min(ip_0, ip_K);
        double info_rot_exact = std::min(ir_0, ir_K);

        global_graph_.update_edge_info(kf_odom_edge_indices_[kf],
                                       info_pos_exact, info_rot_exact);
    }

    // ── Local uncertainty propagation ────────────────────────────────────
    // After convergence, the local graph (including the strong virtual anchor
    // edge) gives the effective information for all nodes in this window.
    // Inverting its anchored Hessian yields the marginal covariance for each
    // fine node from the joint fine measurement + global correction.
    {
        auto local_covs = local.marginal_covariances();
        // Boundary KF nodes get covariance from global graph (below), but set
        // them from the local view for consistency within the window.
        for (size_t li = 0; li < n_local; ++li) {
            size_t fi = (li == n_local - 1) ? fine_end : fine_start + li;
            fine_covariances_[fi] = local_covs[li];
        }
    }

    // ── Write results back ────────────────────────────────────────────────
    const auto& ln = local.nodes();

    // Snap endpoints to the authoritative corrected KF positions.
    fine_nodes_[fine_start].x     = kf_corrected[kf].x;
    fine_nodes_[fine_start].y     = kf_corrected[kf].y;
    fine_nodes_[fine_start].theta = kf_corrected[kf].theta;

    // Copy optimised interior nodes.
    for (size_t li = 1; li < n_local - 1; ++li) {
        fine_nodes_[fine_start + li].x     = ln[li].x;
        fine_nodes_[fine_start + li].y     = ln[li].y;
        fine_nodes_[fine_start + li].theta = ln[li].theta;
    }

    // Snap end KF to authoritative corrected position.
    fine_nodes_[fine_end].x     = kf_corrected[kf + 1].x;
    fine_nodes_[fine_end].y     = kf_corrected[kf + 1].y;
    fine_nodes_[fine_end].theta = kf_corrected[kf + 1].theta;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

double HierarchicalPoseGraphSlam::normalize_angle(double a)
{
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

} // namespace slam
