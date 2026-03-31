#pragma once

#include <cstddef>
#include <vector>

#include "../pose_graph_slam/pose_graph_slam.h"
#include "../utils/matrix.h"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Fine-level pose (one per odometry step)
// ─────────────────────────────────────────────────────────────────────────────

struct FineNode {
    double x, y, theta;
};

// Raw odometry measurement stored for re-propagation after global correction.
struct FineEdge {
    size_t from, to;
    double dx, dy, dtheta;   // in LOCAL frame of 'from' node
};

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalPoseGraphSlam
//
// Two-level online pose graph SLAM.
//
// Level 0 (fine) — every odometry step becomes a FineNode.  Raw measurements
//   are stored in FineEdge so they can be re-rolled from corrected keyframes.
//
// Level 1 (global / coarse) — every `keyframe_interval` steps one FineNode
//   is promoted to a keyframe inside a standard PoseGraphSlam.  The odometry
//   between two consecutive keyframes is *pre-integrated*: the K fine steps
//   are composed (SE(2) chain) into one keyframe-to-keyframe measurement.
//   Loop-closure edges between keyframes also live here.
//   Gauss-Newton runs on this smaller graph; its size is N/K.
//
// Optimization pipeline
// ─────────────────────
//   1. global_graph_.optimize()  — Gauss-Newton on keyframe graph
//   2. propagate_corrections()   — for each keyframe window, snap the
//      keyframe FineNode to its corrected pose, then re-roll the stored
//      fine odometry forward to produce corrected fine-level poses.
//
// Online usage
// ────────────
//   HierarchicalPoseGraphSlam hpg(10);   // keyframe every 10 fine steps
//   for each odometry tick:
//       hpg.add_pose(dx, dy, dtheta);
//   // detect loop closures between keyframes i and j:
//   hpg.add_loop_closure(i, j, dx, dy, dtheta, info_pos, info_rot);
//   hpg.optimize();
//   // corrected dense trajectory:
//   const auto& poses = hpg.fine_nodes();
// ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPoseGraphSlam {
public:
    // keyframe_interval : number of fine steps between consecutive keyframes (K)
    // kf_info_pos/rot   : information weight for pre-integrated keyframe edges
    // fine_info_pos/rot : information (inverse variance) of a single fine
    //   odometry step.  The keyframe-edge information is derived internally
    //   as fine_info / K (linear-chain approximation of the Schur complement).
    explicit HierarchicalPoseGraphSlam(
        size_t keyframe_interval = 10,
        double fine_info_pos     = 1111.0,   // 1 / 0.03^2 (3 cm/step)
        double fine_info_rot     = 10000.0); // 1 / 0.01^2 (~0.6 deg/step)

    // ── Online interface ──────────────────────────────────────────────────────

    // Add the next odometry measurement (called every fine step).
    // dx, dy are in the LOCAL frame of the previous fine pose.
    // Returns the index of the newly created FineNode.
    size_t add_pose(double dx, double dy, double dtheta);

    // Add a loop-closure constraint between two keyframe indices.
    // dx, dy, dtheta must be in the LOCAL frame of keyframe from_kf.
    void add_loop_closure(size_t from_kf, size_t to_kf,
                          double dx, double dy, double dtheta,
                          double info_pos, double info_rot);

    // ── Optimization ──────────────────────────────────────────────────────────

    // 1. Run Gauss-Newton on the global keyframe graph.
    // 2. Propagate corrections to all fine nodes.
    // Optionally records per-iteration keyframe snapshots in kf_iter_history.
    // Returns the number of GN iterations performed.
    int optimize(int max_iterations = 100, double tolerance = 1e-4,
                 std::vector<std::vector<PoseNode>>* kf_iter_history = nullptr);

    // ── Accessors ─────────────────────────────────────────────────────────────

    const std::vector<FineNode>&  fine_nodes()      const { return fine_nodes_; }
    const std::vector<PoseNode>&  keyframe_nodes()  const { return global_graph_.nodes(); }
    const std::vector<PoseEdge>&  keyframe_edges()  const { return global_graph_.edges(); }
    size_t num_keyframes()    const { return global_graph_.nodes().size(); }
    size_t keyframe_interval() const { return keyframe_interval_; }

    // Marginal 3×3 covariance for each keyframe node, computed from the global
    // keyframe Hessian after the last call to optimize().
    // Reflects accumulated measurement uncertainty; does NOT include the anchor
    // node's near-zero variance (node 0 is heavily pinned).
    std::vector<kf::Matrix> keyframe_covariances() const {
        return global_graph_.marginal_covariances();
    }

    // Per-fine-node 3×3 covariance.  Populated after optimize() is called.
    // Interior nodes (within complete windows) come from local window GN;
    // trailing nodes (incomplete window) and the origin are 3×3 zero matrices
    // (covariance not yet computed for those nodes).
    const std::vector<kf::Matrix>& fine_node_covariances() const {
        return fine_covariances_;
    }

    // Fine-node index of keyframe k
    size_t keyframe_fine_index(size_t k) const { return keyframe_fine_indices_[k]; }

private:
    size_t keyframe_interval_;
    double fine_info_pos_;           // information of one fine odometry step
    double fine_info_rot_;
    double kf_info_pos_;             // = fine_info_pos_ / keyframe_interval_
    double kf_info_rot_;             // = fine_info_rot_ / keyframe_interval_

    // ── Fine level ────────────────────────────────────────────────────────────
    std::vector<FineNode> fine_nodes_;   // all fine nodes (grows online)
    std::vector<FineEdge> fine_edges_;   // fine_edges_[i]: fine_nodes_[i] → fine_nodes_[i+1]

    // ── Global keyframe graph ─────────────────────────────────────────────────
    PoseGraphSlam       global_graph_;
    std::vector<size_t> keyframe_fine_indices_;  // keyframe_fine_indices_[k] = fine index of KF k

    // ── Pre-integration accumulator ───────────────────────────────────────────
    // Running relative pose from the last promoted keyframe, expressed in
    // that keyframe's local frame.  Composed step by step via SE(2) chaining.
    double accum_x_{0.0}, accum_y_{0.0}, accum_theta_{0.0};
    size_t steps_in_window_{0};

    // ── Odometry edge tracking ────────────────────────────────────────────────
    // Index (in global_graph_.edges()) of the odometry edge added when each
    // keyframe window completes.  kf_odom_edge_indices_[k] = edge connecting
    // KF_k to KF_{k+1}.  Used by optimize_local_window to update the info
    // matrix with the exact Schur complement after local GN.
    std::vector<size_t> kf_odom_edge_indices_;

    // ── Per-fine-node covariances ─────────────────────────────────────────────
    // 3×3 marginal covariance for each fine node, computed from the local
    // window Hessian.  Entries for the trailing incomplete window and the
    // origin node are 3×3 zero matrices (unavailable).
    std::vector<kf::Matrix> fine_covariances_;

    // ── Helpers ───────────────────────────────────────────────────────────────

    // After global optimization: for each COMPLETE window run a local
    // Gauss-Newton that pins both endpoint keyframes and optimises the
    // interior fine nodes.  The trailing incomplete window (current robot
    // position) falls back to simple dead-reckoning from the corrected anchor.
    void propagate_corrections();

    // Build a temporary PoseGraphSlam for the fine nodes in window [kf, kf+1],
    // pin both KF endpoints, run GN, write corrected positions back.
    void optimize_local_window(size_t kf);

    static double normalize_angle(double a);
};

} // namespace slam
