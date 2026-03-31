// robust_pose_graph_slam_demo.cpp
//
// Robust Pose Graph SLAM demo — Max-Mixture + Huber kernel.
//
// Scenario
// --------
// The robot drives a rectangular loop twice (identical to pose_graph_slam_demo).
// Three kinds of edges are added:
//
//   1. Odometry edges  — standard single-Gaussian, Huber kernel applied.
//
//   2. True loop-closure edges  — MM edges with two components:
//        Component 0 (inlier):  tight Gaussian, weight = 0.9
//        Component 1 (null):    diffuse Gaussian (flat prior), weight = 0.1
//      When the robot revisits a place, the inlier component wins.
//
//   3. False (injected) loop-closure edges  — same MM structure, but the
//      "measurement" is forged to link two distant poses.  The resulting error
//      under the inlier component is huge, so the null/outlier component wins
//      and the edge contributes almost nothing to H and b.
//
// Key output: selected_components() tells us which MM component won per edge.
// Component 0 = inlier treated; component 1 = outlier suppressed.
//
// Outputs (in output/)
//   robust_trajectory.csv     – per-node: true, initial, optimised poses
//   robust_edges.csv          – edges with type and MM classification
//   robust_iterations.csv     – node positions at each GN iteration

#include "../slam_core/robust_pose_graph_slam/robust_pose_graph_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static double normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

struct TruePose { double x, y, theta; };

// ─────────────────────────────────────────────────────────────────────────────
// Build a Max-Mixture loop-closure edge
//
// Component 0 (inlier): tight Gaussian around the supplied measurement.
// Component 1 (null):   diffuse isotropic Gaussian — the "outlier hypothesis".
//   Its information is so low that it always contributes near-zero to H and b;
//   its weight (0.1) is lower so a small inlier error still beats it.
// ─────────────────────────────────────────────────────────────────────────────

static slam::RobustPoseEdge make_mm_lc(
        size_t from, size_t to,
        double dx, double dy, double dtheta,
        double info_pos, double info_rot) {

    // Inlier component
    slam::MMComponent inlier;
    inlier.dx         = dx;
    inlier.dy         = dy;
    inlier.dtheta     = dtheta;
    inlier.info       = kf::Matrix(3, 3, 0.0);
    inlier.info(0, 0) = info_pos;
    inlier.info(1, 1) = info_pos;
    inlier.info(2, 2) = info_rot;
    inlier.log_weight = std::log(0.9);

    // Null / outlier component — very flat, weight = 0.1
    slam::MMComponent null_comp;
    null_comp.dx         = 0.0;
    null_comp.dy         = 0.0;
    null_comp.dtheta     = 0.0;
    null_comp.info       = kf::Matrix(3, 3, 0.0);
    null_comp.info(0, 0) = 1e-4;   // near-zero information = flat prior
    null_comp.info(1, 1) = 1e-4;
    null_comp.info(2, 2) = 1e-4;
    null_comp.log_weight = std::log(0.1);

    return {from, to, {inlier, null_comp}, "loop_closure"};
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::default_random_engine rng(42);

    // Huber delta: χ² threshold that separates inliers from outliers.
    // For a 3-DOF constraint, χ²(95%) ≈ 7.815 → δ ≈ √7.815 ≈ 2.8.
    // We use δ = 1.5 (slightly tighter — keeps robust during early iterations).
    slam::RobustPoseGraphSlam rpg(slam::RobustKernel::Huber, /*delta=*/1.5);

    // ── Noise parameters ─────────────────────────────────────────────────────
    // Keep odometry noise small so the initial dead-reckoning state stays
    // within the inlier basin of the true loop closures (~0.07m per axis
    // after one full loop).  Max-Mixture + Huber then handle whatever
    // residual error remains, and confidently suppress the injected outliers.
    const double odom_lin_std = 0.008;
    const double odom_ang_std = 0.003;
    const double lc_lin_std   = 0.02;
    const double lc_ang_std   = 0.01;

    const double odom_info_pos = 1.0 / (odom_lin_std * odom_lin_std);
    const double odom_info_rot = 1.0 / (odom_ang_std * odom_ang_std);
    const double lc_info_pos   = 1.0 / (lc_lin_std * lc_lin_std);
    const double lc_info_rot   = 1.0 / (lc_ang_std * lc_ang_std);

    // ── True trajectory (same rectangle × 2 as pose_graph_slam_demo) ─────────
    struct WP { double x, y; };
    const std::vector<WP> waypoints = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
        {10.0, 0.0}, {10.0, 7.0}, {0.0, 7.0}, {0.0, 0.0},
    };
    const double step_size = 0.5;

    std::vector<TruePose> true_poses;
    true_poses.push_back({waypoints[0].x, waypoints[0].y, 0.0});

    for (size_t w = 1; w < waypoints.size(); ++w) {
        double sx  = waypoints[w].x - waypoints[w-1].x;
        double sy  = waypoints[w].y - waypoints[w-1].y;
        double len = std::hypot(sx, sy);
        double hdg = std::atan2(sy, sx);
        int ns = std::max(1, static_cast<int>(std::round(len / step_size)));
        double actual = len / ns;
        for (int k = 1; k <= ns; ++k)
            true_poses.push_back({waypoints[w-1].x + k * actual * std::cos(hdg),
                                   waypoints[w-1].y + k * actual * std::sin(hdg),
                                   hdg});
    }

    const size_t num_poses = true_poses.size();

    // ── Build graph with noisy odometry ──────────────────────────────────────
    std::normal_distribution<double> noise_lin(0.0, odom_lin_std);
    std::normal_distribution<double> noise_ang(0.0, odom_ang_std);
    std::normal_distribution<double> noise_lc_lin(0.0, lc_lin_std);
    std::normal_distribution<double> noise_lc_ang(0.0, lc_ang_std);

    std::vector<TruePose> odom_poses(num_poses);
    odom_poses[0] = true_poses[0];
    rpg.add_node(true_poses[0].x, true_poses[0].y, true_poses[0].theta);

    // Track edge metadata for CSV
    struct EdgeMeta { size_t from, to; std::string type; bool is_outlier; };
    std::vector<EdgeMeta> edge_meta;

    for (size_t i = 1; i < num_poses; ++i) {
        double dx_w = true_poses[i].x - true_poses[i-1].x;
        double dy_w = true_poses[i].y - true_poses[i-1].y;
        double dt   = normalize_angle(true_poses[i].theta - true_poses[i-1].theta);

        double ct = std::cos(true_poses[i-1].theta);
        double st = std::sin(true_poses[i-1].theta);
        double dx_l =  ct * dx_w + st * dy_w;
        double dy_l = -st * dx_w + ct * dy_w;

        double m_dx = dx_l + noise_lin(rng);
        double m_dy = dy_l + noise_lin(rng);
        double m_dt = dt   + noise_ang(rng);

        double oct = std::cos(odom_poses[i-1].theta);
        double ost = std::sin(odom_poses[i-1].theta);
        odom_poses[i].x     = odom_poses[i-1].x + oct * m_dx - ost * m_dy;
        odom_poses[i].y     = odom_poses[i-1].y + ost * m_dx + oct * m_dy;
        odom_poses[i].theta = normalize_angle(odom_poses[i-1].theta + m_dt);

        rpg.add_node(odom_poses[i].x, odom_poses[i].y, odom_poses[i].theta);
        rpg.add_edge(i-1, i, m_dx, m_dy, m_dt, odom_info_pos, odom_info_rot,
                     "odometry");
        edge_meta.push_back({i-1, i, "odometry", false});
    }

    // ── True loop closures (MM, inlier component should win) ─────────────────
    const double lc_threshold = 1.5;
    const size_t lc_min_gap   = 20;

    for (size_t i = 0; i < num_poses; ++i) {
        for (size_t j = i + lc_min_gap; j < num_poses; ++j) {
            double dist = std::hypot(true_poses[j].x - true_poses[i].x,
                                     true_poses[j].y - true_poses[i].y);
            if (dist >= lc_threshold) continue;

            double dx_w = true_poses[j].x - true_poses[i].x;
            double dy_w = true_poses[j].y - true_poses[i].y;
            double dt   = normalize_angle(true_poses[j].theta - true_poses[i].theta);
            double ct   = std::cos(true_poses[i].theta);
            double st   = std::sin(true_poses[i].theta);
            double dx_l =  ct * dx_w + st * dy_w + noise_lc_lin(rng);
            double dy_l = -st * dx_w + ct * dy_w + noise_lc_lin(rng);
            double m_dt = dt + noise_lc_ang(rng);

            rpg.add_mm_edge(i, j,
                            make_mm_lc(i, j, dx_l, dy_l, m_dt,
                                       lc_info_pos, lc_info_rot).components,
                            "loop_closure");
            edge_meta.push_back({i, j, "loop_closure", false});
        }
    }

    // ── Injected false (outlier) loop closures ───────────────────────────────
    // Link poses that are geographically far apart — simulating a wrong
    // scan-match or bad place recognition.  The measurement claims they overlap
    // but the error under the inlier component will be huge, so the null
    // component wins and the edge is suppressed.
    const std::vector<std::pair<size_t,size_t>> false_lcs = {
        {5, 80}, {20, 110}, {50, 130}
    };
    for (auto [i, j] : false_lcs) {
        if (i >= num_poses || j >= num_poses) continue;
        // Claim zero relative pose (completely wrong)
        rpg.add_mm_edge(i, j,
                        make_mm_lc(i, j, 0.0, 0.0, 0.0,
                                   lc_info_pos, lc_info_rot).components,
                        "loop_closure");
        edge_meta.push_back({i, j, "loop_closure", true});  // true = intended outlier
    }

    // ── Save initial poses ────────────────────────────────────────────────────
    std::vector<slam::RobPoseNode> initial_poses = rpg.nodes();

    // ── Optimise ──────────────────────────────────────────────────────────────
    std::vector<std::vector<slam::RobPoseNode>> iter_history;
    double err_before = rpg.total_error();
    int iters = rpg.optimize(100, 1e-4, &iter_history);
    double err_after  = rpg.total_error();

    // Per-edge MM classification from the final iteration
    const auto& sel = rpg.selected_components();

    size_t n_inlier = 0, n_outlier = 0;
    for (size_t idx = 0; idx < rpg.edges().size(); ++idx) {
        if (rpg.edges()[idx].tag == "loop_closure") {
            if (sel[idx] == 0) ++n_inlier; else ++n_outlier;
        }
    }

    std::cout << "Poses: " << num_poses
              << " | Edges: "  << rpg.edges().size()
              << " (odom: " << (num_poses - 1)
              << ", LC: " << (rpg.edges().size() - num_poses + 1) << ")\n";
    std::cout << "False LCs injected: " << false_lcs.size() << "\n";
    std::cout << "Gauss-Newton converged in " << iters << " iterations\n";
    std::cout << "Error: " << err_before << " -> " << err_after << "\n";
    std::cout << "MM classification — inlier: " << n_inlier
              << "  outlier (suppressed): " << n_outlier << "\n";

    // ── Write outputs ─────────────────────────────────────────────────────────
    std::filesystem::create_directories("output");

    // Trajectory
    {
        std::ofstream f("output/robust_trajectory.csv");
        f << "node,x_true,y_true,theta_true,"
             "x_init,y_init,theta_init,"
             "x_opt,y_opt,theta_opt\n";
        const auto& opt = rpg.nodes();
        for (size_t i = 0; i < num_poses; ++i)
            f << i << ','
              << true_poses[i].x    << ',' << true_poses[i].y    << ',' << true_poses[i].theta << ','
              << initial_poses[i].x << ',' << initial_poses[i].y << ',' << initial_poses[i].theta << ','
              << opt[i].x           << ',' << opt[i].y           << ',' << opt[i].theta << '\n';
    }

    // Edges with MM classification
    {
        std::ofstream f("output/robust_edges.csv");
        f << "from,to,type,injected_outlier,mm_selected\n";
        size_t lc_idx = num_poses - 1;  // odometry edges come first
        for (size_t idx = 0; idx < edge_meta.size(); ++idx) {
            const auto& em = edge_meta[idx];
            int mm_sel = sel[idx];
            f << em.from << ',' << em.to << ',' << em.type << ','
              << (em.is_outlier ? 1 : 0) << ',' << mm_sel << '\n';
        }
    }

    // Per-iteration snapshots
    {
        std::ofstream f("output/robust_iterations.csv");
        f << "iter,node,x,y,theta\n";
        for (size_t it = 0; it < iter_history.size(); ++it) {
            const auto& snap = iter_history[it];
            for (size_t i = 0; i < snap.size(); ++i)
                f << it << ',' << i << ','
                  << snap[i].x << ',' << snap[i].y << ','
                  << snap[i].theta << '\n';
        }
    }

    std::cout << "Wrote output/robust_trajectory.csv, "
              << "output/robust_edges.csv, "
              << "output/robust_iterations.csv\n";
    return 0;
}
