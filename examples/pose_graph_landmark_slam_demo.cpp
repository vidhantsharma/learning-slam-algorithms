// pose_graph_landmark_slam_demo.cpp
//
// 2-D Pose Graph SLAM with Landmarks demo (Gauss-Newton + LM damping back-end).
//
// Scenario
// --------
// The robot drives a rectangular loop twice.  Noisy odometry links consecutive
// poses.  In addition, the robot observes 2-D landmarks (x,y sensor) that are
// scattered in the environment.  Each landmark observation creates a
// pose-to-landmark edge in the graph.  Because repeated observations of the
// same landmark from different poses implicitly provide loop-closure
// information, the landmark graph can correct odometry drift even without
// explicit pose-to-pose loop-closure edges.
//
// A Levenberg-Marquardt-style damping term λI is added to the Hessian before
// solving  (H + λI) · Δx = −b  to keep the system positive-definite when
// landmarks are observed only once (rank-deficient sub-system).
//
// Outputs (in output/)
//   pg_lm_trajectory.csv    – per-pose: true, initial, optimised x/y/theta
//   pg_lm_landmarks.csv     – per-landmark: true, initial, optimised x/y
//   pg_lm_edges.csv         – edge list with type (odometry | landmark)
//   pg_lm_iterations.csv    – pose + landmark positions at every GN iteration

#include "../slam_core/pose_graph_landmark_slam/pose_graph_landmark_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

static double normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

struct TruePose { double x, y, theta; };
struct TrueLM   { double x, y; };

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::default_random_engine rng(7);

    // ── Noise parameters ─────────────────────────────────────────────────────
    const double odom_lin_std = 0.05;      // 5 cm per step
    const double odom_ang_std = 0.015;     // ~0.9° per step
    const double obs_std      = 0.10;      // 10 cm landmark observation noise
    const double obs_range    = 5.0;       // maximum detection range (m)

    const double odom_info_pos = 1.0 / (odom_lin_std * odom_lin_std);
    const double odom_info_rot = 1.0 / (odom_ang_std * odom_ang_std);
    const double obs_info      = 1.0 / (obs_std * obs_std);

    // ── True landmarks (scattered around the loop) ───────────────────────────
    std::vector<TrueLM> true_lms = {
        { 2.0,  1.5}, { 5.0,  0.5}, { 8.5,  1.0},
        { 9.5,  3.5}, { 8.0,  6.0}, { 5.5,  6.5},
        { 2.0,  6.5}, { 0.5,  4.0}, { 0.5,  1.5},
        { 5.0,  3.5}, { 3.0,  4.5}, { 7.0,  4.0},
    };
    const size_t num_lms = true_lms.size();

    // ── True trajectory: robot drives a rectangle twice ──────────────────────
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
        for (int k = 1; k <= ns; ++k) {
            double px = waypoints[w-1].x + k * actual * std::cos(hdg);
            double py = waypoints[w-1].y + k * actual * std::sin(hdg);
            true_poses.push_back({px, py, hdg});
        }
    }
    const size_t num_poses = true_poses.size();

    // ── Build graph with noisy odometry ──────────────────────────────────────
    slam::PoseGraphLandmarkSlam pg;

    std::normal_distribution<double> noise_lin(0.0, odom_lin_std);
    std::normal_distribution<double> noise_ang(0.0, odom_ang_std);
    std::normal_distribution<double> noise_obs(0.0, obs_std);

    // Add first pose (known start) and landmarks with no-observation initial
    // estimate (will be set from the first observation below).
    pg.add_pose(true_poses[0].x, true_poses[0].y, true_poses[0].theta);
    for (size_t l = 0; l < num_lms; ++l)
        pg.add_landmark(0.0, 0.0);    // placeholder; initial estimate below

    // Tracks per-landmark initial estimate and whether we've seen it yet.
    std::vector<bool>   lm_seen(num_lms, false);
    std::vector<double> lm_init_x(num_lms, 0.0), lm_init_y(num_lms, 0.0);

    // Dead-reckoned poses
    std::vector<TruePose> odom_poses(num_poses);
    odom_poses[0] = true_poses[0];

    struct EdgeInfo { size_t from, to; std::string type; };
    std::vector<EdgeInfo> edge_log;

    for (size_t i = 1; i < num_poses; ++i) {
        // True relative motion from i-1 to i (in frame i-1)
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

        // Dead-reckon
        double oct = std::cos(odom_poses[i-1].theta);
        double ost = std::sin(odom_poses[i-1].theta);
        odom_poses[i].x     = odom_poses[i-1].x + oct * m_dx - ost * m_dy;
        odom_poses[i].y     = odom_poses[i-1].y + ost * m_dx + oct * m_dy;
        odom_poses[i].theta = normalize_angle(odom_poses[i-1].theta + m_dt);

        pg.add_pose(odom_poses[i].x, odom_poses[i].y, odom_poses[i].theta);
        pg.add_odometry_edge(i-1, i, m_dx, m_dy, m_dt,
                             odom_info_pos, odom_info_rot);
        edge_log.push_back({i-1, i, "odometry"});

        // ── Landmark observations from pose i ─────────────────────────────
        const TruePose& tp = true_poses[i];
        for (size_t l = 0; l < num_lms; ++l) {
            double dist = std::hypot(true_lms[l].x - tp.x,
                                     true_lms[l].y - tp.y);
            if (dist > obs_range)
                continue;

            // True relative observation (robot frame at true pose)
            double dlx = true_lms[l].x - tp.x;
            double dly = true_lms[l].y - tp.y;
            double ctt = std::cos(tp.theta);
            double stt = std::sin(tp.theta);
            double zx  =  ctt * dlx + stt * dly + noise_obs(rng);
            double zy  = -stt * dlx + ctt * dly + noise_obs(rng);

            pg.add_landmark_edge(i, l, zx, zy, obs_info);
            edge_log.push_back({i, l, "landmark"});

            // Set landmark initial estimate from first observation
            if (!lm_seen[l]) {
                lm_seen[l] = true;
                // project measured observation forward using the odom pose
                double oct2 = std::cos(odom_poses[i].theta);
                double ost2 = std::sin(odom_poses[i].theta);
                lm_init_x[l] = odom_poses[i].x + oct2 * zx - ost2 * zy;
                lm_init_y[l] = odom_poses[i].y + ost2 * zx + oct2 * zy;
            }
        }
    }

    // Populate landmark initial estimates
    for (size_t l = 0; l < num_lms; ++l) {
        if (!lm_seen[l]) {
            // Fallback if landmark was never observed
            lm_init_x[l] = true_lms[l].x;
            lm_init_y[l] = true_lms[l].y;
        }
    }
    // Re-initialise landmarks in the graph using the first-observation estimate
    // (the graph's landmark vector is accessible; we rebuild it properly)
    for (size_t l = 0; l < num_lms; ++l) {
        // Access mutable landmark through re-add (the graph stores them as a
        // plain vector; we manipulate data via public const accessor + cast)
        // Instead: replace the stored x/y by re-using the add sequence.
        // Since add_landmark pushes to a vector we can't directly set
        // after-the-fact via the const API, we use a small trick:
        // store in a separate structure and set via the snapshot mechanism.
    }

    // ── Save a snapshot of initial estimates then override with first-obs ─────
    // Retrieve current initial poses (dead-reckoned)
    std::vector<slam::PLPoseNode>     initial_poses     = pg.poses();
    // Override landmarks with first-observation-based initial estimates
    // by storing them before optimization (the optimize() will update them).
    std::vector<slam::PLLandmarkNode> initial_lms(num_lms);
    for (size_t l = 0; l < num_lms; ++l)
        initial_lms[l] = {lm_init_x[l], lm_init_y[l]};

    // ── Set landmark initial positions in the graph ───────────────────────────
    // We do this by running one iteration starting from the first-obs estimate.
    // Since PoseGraphLandmarkSlam stores landmarks_ as a non-const member we
    // access them through a const_cast (safe – we own the object).
    {
        auto& lm_vec = const_cast<std::vector<slam::PLLandmarkNode>&>(pg.landmarks());
        for (size_t l = 0; l < num_lms; ++l)
            lm_vec[l] = initial_lms[l];
    }

    // ── Optimise ──────────────────────────────────────────────────────────────
    std::vector<slam::PLSnapshot> iter_history;
    double err_before = pg.total_error();
    int iters = pg.optimize(200, 1e-5, 1e-4, &iter_history);
    double err_after  = pg.total_error();

    std::cout << "Poses: "     << num_poses
              << " | Landmarks: " << num_lms
              << " | Odom edges: " << (num_poses - 1)
              << " | LM edges: "   << pg.lm_edges().size() << "\n";
    std::cout << "Gauss-Newton converged in " << iters << " iterations\n";
    std::cout << "Error: " << err_before << " -> " << err_after << "\n";

    // ── Write outputs ─────────────────────────────────────────────────────────
    std::filesystem::create_directories("output");

    // Trajectory
    {
        std::ofstream f("output/pg_lm_trajectory.csv");
        f << "node,x_true,y_true,theta_true,"
             "x_init,y_init,theta_init,"
             "x_opt,y_opt,theta_opt\n";
        const auto& opt = pg.poses();
        for (size_t i = 0; i < num_poses; ++i) {
            f << i << ','
              << true_poses[i].x      << ',' << true_poses[i].y      << ',' << true_poses[i].theta     << ','
              << initial_poses[i].x   << ',' << initial_poses[i].y   << ',' << initial_poses[i].theta  << ','
              << opt[i].x             << ',' << opt[i].y             << ',' << opt[i].theta             << '\n';
        }
    }

    // Landmarks
    {
        std::ofstream f("output/pg_lm_landmarks.csv");
        f << "lm_id,x_true,y_true,x_init,y_init,x_opt,y_opt\n";
        const auto& opt_lms = pg.landmarks();
        for (size_t l = 0; l < num_lms; ++l) {
            f << l << ','
              << true_lms[l].x     << ',' << true_lms[l].y     << ','
              << initial_lms[l].x  << ',' << initial_lms[l].y  << ','
              << opt_lms[l].x      << ',' << opt_lms[l].y      << '\n';
        }
    }

    // Edges
    {
        std::ofstream f("output/pg_lm_edges.csv");
        f << "from,to,type\n";
        for (const auto& e : edge_log)
            f << e.from << ',' << e.to << ',' << e.type << '\n';
    }

    // Per-iteration snapshots
    {
        std::ofstream f("output/pg_lm_iterations.csv");
        f << "iter,type,idx,x,y\n";
        for (size_t it = 0; it < iter_history.size(); ++it) {
            const auto& snap = iter_history[it];
            for (size_t i = 0; i < snap.poses.size(); ++i)
                f << it << ",pose," << i << ','
                  << snap.poses[i].x << ',' << snap.poses[i].y << '\n';
            for (size_t l = 0; l < snap.landmarks.size(); ++l)
                f << it << ",landmark," << l << ','
                  << snap.landmarks[l].x << ',' << snap.landmarks[l].y << '\n';
        }
    }

    std::cout << "Wrote output/pg_lm_trajectory.csv, output/pg_lm_landmarks.csv, "
              << "output/pg_lm_edges.csv, output/pg_lm_iterations.csv\n";
    return 0;
}
