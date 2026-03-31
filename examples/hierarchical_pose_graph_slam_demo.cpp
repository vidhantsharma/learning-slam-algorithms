// hierarchical_pose_graph_slam_demo.cpp
//
// Two-level Hierarchical Pose Graph SLAM demo.
//
// Scenario
// --------
// Same rectangular double loop as the flat pose graph demo.  Noisy odometry
// links every fine step.  Every K=10 fine steps a keyframe is promoted.
// Loop closures are detected between keyframes (proximity of ground-truth
// keyframe positions).  Gauss-Newton runs on the small keyframe graph;
// corrections then propagate down to all fine nodes.
//
//  Flat graph size  : N nodes, N-1 odometry edges + loop-closure edges
//  Keyframe graph   : N/K keyframe nodes (much smaller → faster optimization)
//
// Outputs (in output/)
//   hierarchical_trajectory.csv    – fine nodes: true / initial / optimised
//   hierarchical_kf_edges.csv      – keyframe-level edges with type
//   hierarchical_iterations.csv    – keyframe positions per GN iteration

#include "../slam_core/hierarchical_pose_graph_slam/hierarchical_pose_graph_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

struct TruePose { double x, y, theta; };

static double normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::default_random_engine rng(42);

    // ── Noise parameters ─────────────────────────────────────────────────────
    const double odom_lin_std  = 0.03;
    const double odom_ang_std  = 0.01;
    const double lc_lin_std    = 0.05;
    const double lc_ang_std    = 0.02;

    const double odom_info_pos = 1.0 / (odom_lin_std * odom_lin_std);
    const double odom_info_rot = 1.0 / (odom_ang_std * odom_ang_std);
    const double lc_info_pos   = 1.0 / (lc_lin_std * lc_lin_std);
    const double lc_info_rot   = 1.0 / (lc_ang_std * lc_ang_std);

    // ── Hierarchical parameters ───────────────────────────────────────────────
    // K = 10: one keyframe every 10 fine steps (every ~5 m at 0.5 m/step).
    // The class derives kf_info = odom_info / K internally and uses odom_info
    // for the fine-level local window optimisation.
    const size_t K = 10;

    // ── Ground-truth rectangular double loop ─────────────────────────────────
    struct WP { double x, y; };
    const std::vector<WP> waypoints = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
        {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
    };
    const double step_size = 0.5;   // metres per fine step

    std::vector<TruePose> true_poses;
    true_poses.push_back({waypoints[0].x, waypoints[0].y, 0.0});
    for (size_t w = 1; w < waypoints.size(); ++w) {
        double sx = waypoints[w].x - waypoints[w - 1].x;
        double sy = waypoints[w].y - waypoints[w - 1].y;
        double seg_len = std::hypot(sx, sy);
        double heading = std::atan2(sy, sx);
        int n_steps = static_cast<int>(std::round(seg_len / step_size));
        if (n_steps < 1) n_steps = 1;
        double actual_step = seg_len / n_steps;
        for (int k = 1; k <= n_steps; ++k) {
            double px = waypoints[w - 1].x + k * actual_step * std::cos(heading);
            double py = waypoints[w - 1].y + k * actual_step * std::sin(heading);
            true_poses.push_back({px, py, heading});
        }
    }
    const size_t num_fine = true_poses.size();

    // ── Build hierarchical pose graph with noisy odometry (online) ────────────
    // add_pose() is called for every fine step; the class promotes keyframes
    // automatically whenever K steps have accumulated.
    slam::HierarchicalPoseGraphSlam hpg(K, odom_info_pos, odom_info_rot);

    std::normal_distribution<double> noise_lin(0.0, odom_lin_std);
    std::normal_distribution<double> noise_ang(0.0, odom_ang_std);
    std::normal_distribution<double> noise_lc_lin(0.0, lc_lin_std);
    std::normal_distribution<double> noise_lc_ang(0.0, lc_ang_std);

    for (size_t i = 1; i < num_fine; ++i) {
        // True relative motion from step i-1 to i (rotate into frame of i-1)
        double dx_w = true_poses[i].x - true_poses[i - 1].x;
        double dy_w = true_poses[i].y - true_poses[i - 1].y;
        double dt   = normalize_angle(true_poses[i].theta - true_poses[i - 1].theta);
        double ct   = std::cos(true_poses[i - 1].theta);
        double st   = std::sin(true_poses[i - 1].theta);

        // Noisy odometry measurement (just one fine step)
        double m_dx = ct * dx_w + st * dy_w + noise_lin(rng);
        double m_dy = -st * dx_w + ct * dy_w + noise_lin(rng);
        double m_dt = dt + noise_ang(rng);

        hpg.add_pose(m_dx, m_dy, m_dt);
    }

    // ── Save initial (dead-reckoned) fine poses before any optimization ───────
    std::vector<slam::FineNode> initial_fine = hpg.fine_nodes();

    // ── Detect keyframe-level loop closures ───────────────────────────────────
    // Use ground-truth proximity of keyframe positions as a proxy for
    // scan-matching detection (same strategy as the flat demo).
    const size_t num_kf       = hpg.num_keyframes();
    const double lc_threshold = 1.5;    // metres
    const size_t lc_min_gap   = 3;      // keyframes apart (= 3 * K fine steps)

    struct KfEdgeInfo { size_t from, to; std::string type; };
    std::vector<KfEdgeInfo> kf_edge_log;

    // Record odometry keyframe edges (already inside the global graph)
    for (size_t k = 0; k + 1 < num_kf; ++k)
        kf_edge_log.push_back({k, k + 1, "odometry"});

    for (size_t i = 0; i < num_kf; ++i) {
        for (size_t j = i + lc_min_gap; j < num_kf; ++j) {
            // True positions of keyframe i and keyframe j
            size_t fi = hpg.keyframe_fine_index(i);
            size_t fj = hpg.keyframe_fine_index(j);
            double dist = std::hypot(true_poses[fj].x - true_poses[fi].x,
                                     true_poses[fj].y - true_poses[fi].y);
            if (dist < lc_threshold) {
                // Relative pose of KF j in the LOCAL frame of KF i, + noise
                double dx_w = true_poses[fj].x - true_poses[fi].x;
                double dy_w = true_poses[fj].y - true_poses[fi].y;
                double dt   = normalize_angle(true_poses[fj].theta -
                                              true_poses[fi].theta);
                double ct = std::cos(true_poses[fi].theta);
                double st = std::sin(true_poses[fi].theta);
                double dx_local =  ct * dx_w + st * dy_w + noise_lc_lin(rng);
                double dy_local = -st * dx_w + ct * dy_w + noise_lc_lin(rng);
                double m_dt = dt + noise_lc_ang(rng);

                hpg.add_loop_closure(i, j, dx_local, dy_local, m_dt,
                                     lc_info_pos, lc_info_rot);
                kf_edge_log.push_back({i, j, "loop_closure"});
            }
        }
    }

    // ── Optimize ──────────────────────────────────────────────────────────────
    std::vector<std::vector<slam::PoseNode>> kf_iter_history;
    int iters = hpg.optimize(100, 1e-4, &kf_iter_history);

    const std::vector<slam::FineNode>& opt_fine = hpg.fine_nodes();
    const size_t n_lc = hpg.keyframe_edges().size() - (num_kf - 1);

    std::cout << "Fine nodes  : " << num_fine << "\n";
    std::cout << "Keyframes   : " << num_kf
              << "  (interval K=" << K << ")\n";
    std::cout << "KF edges    : " << hpg.keyframe_edges().size()
              << "  (odometry: " << (num_kf - 1)
              << ", loop-closure: " << n_lc << ")\n";
    std::cout << "GN iterations: " << iters << "\n";

    // ── Write outputs ─────────────────────────────────────────────────────────
    std::filesystem::create_directories("output");

    // Fine trajectory: true / initial / optimised + keyframe flag
    {
        std::ofstream f("output/hierarchical_trajectory.csv");
        f << "node,x_true,y_true,theta_true,"
             "x_init,y_init,theta_init,"
             "x_opt,y_opt,theta_opt,is_keyframe\n";

        std::vector<bool> is_kf(num_fine, false);
        for (size_t k = 0; k < num_kf; ++k)
            is_kf[hpg.keyframe_fine_index(k)] = true;

        for (size_t i = 0; i < num_fine; ++i) {
            f << i << ','
              << true_poses[i].x     << ',' << true_poses[i].y     << ',' << true_poses[i].theta     << ','
              << initial_fine[i].x   << ',' << initial_fine[i].y   << ',' << initial_fine[i].theta   << ','
              << opt_fine[i].x       << ',' << opt_fine[i].y       << ',' << opt_fine[i].theta       << ','
              << (is_kf[i] ? 1 : 0) << '\n';
        }
    }

    // Keyframe-level edges
    {
        std::ofstream f("output/hierarchical_kf_edges.csv");
        f << "from_kf,to_kf,type\n";
        for (const auto& e : kf_edge_log)
            f << e.from << ',' << e.to << ',' << e.type << '\n';
    }

    // Keyframe positions at each GN iteration (for animation)
    {
        std::ofstream f("output/hierarchical_iterations.csv");
        f << "iter,kf,x,y,theta\n";
        for (size_t it = 0; it < kf_iter_history.size(); ++it) {
            const auto& snap = kf_iter_history[it];
            for (size_t k = 0; k < snap.size(); ++k) {
                f << it << ',' << k << ','
                  << snap[k].x << ',' << snap[k].y << ',' << snap[k].theta << '\n';
            }
        }
    }

    std::cout << "Wrote output/hierarchical_trajectory.csv, "
              << "output/hierarchical_kf_edges.csv, "
              << "output/hierarchical_iterations.csv\n";
    return 0;
}
