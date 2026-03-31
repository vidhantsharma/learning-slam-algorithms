// pose_graph_slam_demo.cpp
//
// 2-D Pose Graph SLAM demo (Gauss-Newton back-end).
//
// Scenario
// --------
// The robot drives a rectangular loop twice inside a simulated environment.
// Noisy odometry links consecutive poses (odometry edges).  When the robot
// revisits a location close to a previous pose, a loop-closure edge is added
// (simulating ICP / scan-matching front-end).
//
// The demo saves:
//   1. poses before optimization  (dead-reckoned / odometry-only)
//   2. poses after Gauss-Newton optimization
//   3. ground-truth poses
//   4. edges (odometry + loop-closure)
//   5. per-iteration pose snapshots (for animation)
//
// Outputs (in output/)
//   pose_graph_trajectory.csv   – per-node: true, initial, optimised poses
//   pose_graph_edges.csv        – all edges with type (odometry / loop_closure)
//   pose_graph_iterations.csv   – node positions at each GN iteration

#include "../slam_core/pose_graph_slam/pose_graph_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Helper types
// ─────────────────────────────────────────────────────────────────────────────

struct TruePose {
    double x, y, theta;
};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

static int parse_steps(int argc, char** argv, int def = 200) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc) return std::stoi(argv[i + 1]);
    }
    return def;
}

// ─────────────────────────────────────────────────────────────────────────────
// Angle utilities
// ─────────────────────────────────────────────────────────────────────────────

static double normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    (void)parse_steps(argc, argv);   // accepted but unused (fixed trajectory)

    std::default_random_engine rng(42);

    // ── Noise parameters ─────────────────────────────────────────────────────
    const double odom_lin_std   = 0.03;     // 3 cm per step
    const double odom_ang_std   = 0.01;     // ~0.6° per step
    const double lc_lin_std     = 0.05;     // loop-closure position noise
    const double lc_ang_std     = 0.02;     // loop-closure angle noise

    // Information (inverse variance) for edges
    const double odom_info_pos  = 1.0 / (odom_lin_std * odom_lin_std);
    const double odom_info_rot  = 1.0 / (odom_ang_std * odom_ang_std);
    const double lc_info_pos    = 1.0 / (lc_lin_std * lc_lin_std);
    const double lc_info_rot    = 1.0 / (lc_ang_std * lc_ang_std);

    // ── Define the true rectangular trajectory ───────────────────────────────
    // Waypoints forming a rectangle that the robot drives around twice.
    struct WP { double x, y; };
    const std::vector<WP> waypoints = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
        {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
    };
    const double step_size = 0.5;  // metres per step

    // Generate ground-truth poses along the path
    std::vector<TruePose> true_poses;
    true_poses.push_back({waypoints[0].x, waypoints[0].y, 0.0});

    for (size_t w = 1; w < waypoints.size(); ++w) {
        double sx = waypoints[w].x - waypoints[w - 1].x;
        double sy = waypoints[w].y - waypoints[w - 1].y;
        double seg_len = std::hypot(sx, sy);
        double heading = std::atan2(sy, sx);
        int    n_steps = static_cast<int>(std::round(seg_len / step_size));
        if (n_steps < 1) n_steps = 1;
        double actual_step = seg_len / n_steps;

        for (int k = 1; k <= n_steps; ++k) {
            double px = waypoints[w - 1].x + k * actual_step * std::cos(heading);
            double py = waypoints[w - 1].y + k * actual_step * std::sin(heading);
            true_poses.push_back({px, py, heading});
        }
    }

    const size_t num_poses = true_poses.size();

    // ── Build pose graph with noisy odometry ─────────────────────────────────
    slam::PoseGraphSlam pg;
    std::normal_distribution<double> noise_lin(0.0, odom_lin_std);
    std::normal_distribution<double> noise_ang(0.0, odom_ang_std);
    std::normal_distribution<double> noise_lc_lin(0.0, lc_lin_std);
    std::normal_distribution<double> noise_lc_ang(0.0, lc_ang_std);

    // Dead-reckoned poses (accumulated noisy odometry)
    std::vector<TruePose> odom_poses(num_poses);
    odom_poses[0] = true_poses[0];      // known start
    pg.add_node(true_poses[0].x, true_poses[0].y, true_poses[0].theta);

    // Edge type tracking for CSV output
    struct EdgeInfo { size_t from, to; std::string type; };
    std::vector<EdgeInfo> edge_log;

    for (size_t i = 1; i < num_poses; ++i) {
        // True relative motion from i-1 to i
        double dx_w = true_poses[i].x - true_poses[i - 1].x;
        double dy_w = true_poses[i].y - true_poses[i - 1].y;
        double dt   = normalize_angle(true_poses[i].theta -
                                       true_poses[i - 1].theta);

        // Rotate into frame of pose i-1
        double ct = std::cos(true_poses[i - 1].theta);
        double st = std::sin(true_poses[i - 1].theta);
        double dx_local =  ct * dx_w + st * dy_w;
        double dy_local = -st * dx_w + ct * dy_w;

        // Add noise → measured odometry
        double m_dx = dx_local + noise_lin(rng);
        double m_dy = dy_local + noise_lin(rng);
        double m_dt = dt       + noise_ang(rng);

        // Dead-reckon: accumulate noisy motion in world frame
        double odom_ct = std::cos(odom_poses[i - 1].theta);
        double odom_st = std::sin(odom_poses[i - 1].theta);
        odom_poses[i].x     = odom_poses[i - 1].x + odom_ct * m_dx - odom_st * m_dy;
        odom_poses[i].y     = odom_poses[i - 1].y + odom_st * m_dx + odom_ct * m_dy;
        odom_poses[i].theta = normalize_angle(odom_poses[i - 1].theta + m_dt);

        pg.add_node(odom_poses[i].x, odom_poses[i].y, odom_poses[i].theta);
        pg.add_edge(i - 1, i, m_dx, m_dy, m_dt, odom_info_pos, odom_info_rot);
        edge_log.push_back({i - 1, i, "odometry"});
    }

    // ── Detect loop closures ─────────────────────────────────────────────────
    // When the robot is near a previously visited pose (using ground truth
    // proximity as a proxy for a front-end scan-matching detection).
    const double lc_threshold = 1.5;      // metres
    const size_t lc_min_gap   = 20;       // minimum index gap

    for (size_t i = 0; i < num_poses; ++i) {
        for (size_t j = i + lc_min_gap; j < num_poses; ++j) {
            double dist = std::hypot(true_poses[j].x - true_poses[i].x,
                                     true_poses[j].y - true_poses[i].y);
            if (dist < lc_threshold) {
                // Compute relative pose from ground truth + noise
                double dx_w = true_poses[j].x - true_poses[i].x;
                double dy_w = true_poses[j].y - true_poses[i].y;
                double dt   = normalize_angle(true_poses[j].theta -
                                               true_poses[i].theta);
                double ct = std::cos(true_poses[i].theta);
                double st = std::sin(true_poses[i].theta);
                double dx_local =  ct * dx_w + st * dy_w + noise_lc_lin(rng);
                double dy_local = -st * dx_w + ct * dy_w + noise_lc_lin(rng);
                double m_dt     = dt + noise_lc_ang(rng);

                pg.add_edge(i, j, dx_local, dy_local, m_dt,
                            lc_info_pos, lc_info_rot);
                edge_log.push_back({i, j, "loop_closure"});
            }
        }
    }

    // ── Save initial (before-optimisation) poses ─────────────────────────────
    std::vector<slam::PoseNode> initial_poses = pg.nodes();

    // ── Optimise ─────────────────────────────────────────────────────────────
    std::vector<std::vector<slam::PoseNode>> iter_history;
    double err_before = pg.total_error();
    int iters = pg.optimize(100, 1e-4, &iter_history);
    double err_after  = pg.total_error();

    std::cout << "Poses: " << num_poses
              << " | Edges: " << pg.edges().size()
              << " (odometry: " << (num_poses - 1)
              << ", loop-closure: " << (pg.edges().size() - num_poses + 1) << ")\n";
    std::cout << "Gauss-Newton converged in " << iters << " iterations\n";
    std::cout << "Error: " << err_before << " -> " << err_after << "\n";

    // ── Write outputs ────────────────────────────────────────────────────────
    std::filesystem::create_directories("output");

    // Trajectory: true, initial (odom), optimised
    {
        std::ofstream f("output/pose_graph_trajectory.csv");
        f << "node,x_true,y_true,theta_true,"
             "x_init,y_init,theta_init,"
             "x_opt,y_opt,theta_opt\n";
        const auto& opt = pg.nodes();
        for (size_t i = 0; i < num_poses; ++i) {
            f << i << ','
              << true_poses[i].x     << ',' << true_poses[i].y     << ',' << true_poses[i].theta << ','
              << initial_poses[i].x  << ',' << initial_poses[i].y  << ',' << initial_poses[i].theta << ','
              << opt[i].x            << ',' << opt[i].y            << ',' << opt[i].theta << '\n';
        }
    }

    // Edges
    {
        std::ofstream f("output/pose_graph_edges.csv");
        f << "from,to,type\n";
        for (const auto& e : edge_log) {
            f << e.from << ',' << e.to << ',' << e.type << '\n';
        }
    }

    // Per-iteration snapshots (for optimisation animation)
    {
        std::ofstream f("output/pose_graph_iterations.csv");
        f << "iter,node,x,y,theta\n";
        for (size_t it = 0; it < iter_history.size(); ++it) {
            const auto& snap = iter_history[it];
            for (size_t i = 0; i < snap.size(); ++i) {
                f << it << ',' << i << ','
                  << snap[i].x << ',' << snap[i].y << ','
                  << snap[i].theta << '\n';
            }
        }
    }

    std::cout << "Wrote output/pose_graph_trajectory.csv, "
              << "output/pose_graph_edges.csv, "
              << "output/pose_graph_iterations.csv\n";
    return 0;
}
