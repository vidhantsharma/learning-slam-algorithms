// grid_slam_demo.cpp
//
// 2-D RBPF (FastSLAM 2.0 / GMapping style) demo.
//
// Scenario
// --------
// The robot drives a random coverage trajectory inside a box room with
// internal walls.  At each step it takes a 180-degree laser scan.
// The RBPF integrates the scans into per-particle occupancy grids.
//
// Outputs (in output/)
//   grid_slam_trajectory.csv   – true + best-particle pose per step
//   grid_slam_particles.csv    – particle cloud (x, y, θ, weight) per step
//   grid_slam_map.csv          – best-particle SLAM map (col, row, log_odds, prob)
//   grid_slam_gt_map.csv       – ground-truth map rasterised from room geometry

#include "../slam_core/grid_slam/grid_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Simple 2-D box-room represented as axis-aligned line segments.
// The laser scanner finds the nearest intersection along each beam.
// ─────────────────────────────────────────────────────────────────────────────

struct Segment {
    double x1, y1, x2, y2;
};

static std::vector<Segment> create_room() {
    // A 20 x 20 m box with several internal wall segments
    return {
        // outer walls
        {-10.0, -10.0,  10.0, -10.0},   // bottom
        { 10.0, -10.0,  10.0,  10.0},   // right
        { 10.0,  10.0, -10.0,  10.0},   // top
        {-10.0,  10.0, -10.0, -10.0},   // left
        // internal obstacles
        {  2.0,  -6.0,   2.0,   4.0},   // vertical wall
        { -6.0,   2.0,   0.0,   2.0},   // horizontal wall
        { -3.0,  -5.0,  -3.0,  -1.0},   // short vertical
        {  5.0,   3.0,   9.0,   3.0},   // horizontal near right
    };
}

// Ray – segment intersection; returns distance or +inf if none.
static double ray_segment_intersect(double rx, double ry, double ra,
                                    const Segment& s) {
    // Parametric: P = (rx,ry) + t*(cos(ra),sin(ra))
    //             Q = (s.x1,s.y1) + u*((s.x2-s.x1),(s.y2-s.y1))
    double dx = std::cos(ra), dy = std::sin(ra);
    double ex = s.x2 - s.x1, ey = s.y2 - s.y1;
    double denom = dx * ey - dy * ex;
    if (std::fabs(denom) < 1e-12) return std::numeric_limits<double>::infinity();
    double t = ((s.x1 - rx) * ey - (s.y1 - ry) * ex) / denom;
    double u = ((s.x1 - rx) * dy - (s.y1 - ry) * dx) / denom;
    if (t < 0.0 || u < 0.0 || u > 1.0) return std::numeric_limits<double>::infinity();
    return t;
}

static slam::LaserScan simulate_scan(double rx, double ry, double rtheta,
                                     const std::vector<Segment>& room,
                                     int num_beams,
                                     double range_max,
                                     std::default_random_engine& rng,
                                     double range_noise_std) {
    slam::LaserScan scan;
    scan.range_max = range_max;
    scan.angle_min = -M_PI;        // full 360-degree scan
    scan.angle_max =  M_PI;
    scan.ranges.resize(static_cast<size_t>(num_beams));

    std::normal_distribution<double> noise(0.0, range_noise_std);
    double step = (num_beams > 1) ? (scan.angle_max - scan.angle_min)
                                    / static_cast<double>(num_beams - 1)
                                  : 0.0;

    for (int i = 0; i < num_beams; ++i) {
        double angle = rtheta + scan.angle_min + i * step;
        double dist  = range_max;
        for (const auto& seg : room) {
            double d = ray_segment_intersect(rx, ry, angle, seg);
            if (d < dist) dist = d;
        }
        scan.ranges[static_cast<size_t>(i)] = dist + noise(rng);
    }
    return scan;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a ground-truth occupancy grid by rasterising the room segments.
// A cell is marked occupied if a wall segment passes through it.
// ─────────────────────────────────────────────────────────────────────────────
static slam::OccupancyGrid build_gt_map(const std::vector<Segment>& room,
                                         double resolution,
                                         double grid_half) {
    slam::OccupancyGrid g;
    g.resolution = resolution;
    int cells    = static_cast<int>(std::ceil(2.0 * grid_half / resolution));
    g.width      = cells;
    g.height     = cells;
    g.origin_x   = -grid_half;
    g.origin_y   = -grid_half;
    g.log_odds.assign(static_cast<size_t>(cells * cells), 0.0);

    // Rasterise each segment using many sample points along it
    for (const auto& s : room) {
        double len   = std::hypot(s.x2 - s.x1, s.y2 - s.y1);
        int    steps = static_cast<int>(std::ceil(len / (resolution * 0.5)));
        for (int k = 0; k <= steps; ++k) {
            double t  = static_cast<double>(k) / static_cast<double>(steps);
            double wx = s.x1 + t * (s.x2 - s.x1);
            double wy = s.y1 + t * (s.y2 - s.y1);
            int col, row;
            g.world_to_cell(wx, wy, col, row);
            if (g.in_bounds(col, row)) {
                g.at(col, row) = 5.0;  // mark as occupied (high log-odds)
            }
        }
    }
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Line-of-sight check: returns true if the segment from (x0,y0) to (x1,y1)
// does NOT cross any wall segment (with a safety margin).
// ─────────────────────────────────────────────────────────────────────────────
static bool line_of_sight(double x0, double y0, double x1, double y1,
                           const std::vector<Segment>& walls,
                           double safety = 0.4) {
    // Cast the ray from (x0,y0) toward (x1,y1).  If the nearest intersection
    // is farther than the target distance (minus safety), the path is clear.
    double desired_angle = std::atan2(y1 - y0, x1 - x0);
    double dist_to_wp    = std::hypot(x1 - x0, y1 - y0);
    for (const auto& seg : walls) {
        double d = ray_segment_intersect(x0, y0, desired_angle, seg);
        if (d < dist_to_wp - safety) return false;
    }
    return true;
}

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
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const int    steps         = parse_steps(argc, argv, 400);
    const size_t num_particles = 30;
    const int    num_beams     = 180;       // full 360° / 2° resolution
    const double range_max     = 12.0;
    const double resolution    = 0.2;       // grid cell size in metres
    const double grid_half     = 13.0;      // half-size of map in metres
    const double room_bound    = 8.5;       // steer-back boundary (< outer wall at 10)
    const int    snap_every    = 20;        // dump map snapshot every N steps

    // Motion parameters
    const double v_base = 0.5;              // linear velocity (m/s)
    const double dt     = 0.2;             // time step (s)

    std::default_random_engine rng(42);
    const std::vector<Segment> room = create_room();

    // ── Pre-defined open-space waypoints ─────────────────────────────────────
    // Hand-picked to lie in the open corridors of the room, away from all walls.
    // The robot cycles through these in order, picking any that has a clear
    // line-of-sight from the current position.
    const std::vector<std::pair<double,double>> waypoints = {
        {  0.0,  0.0}, {  6.0,  0.0}, {  6.0,  6.0}, {  0.0,  6.0},
        { -6.0,  6.0}, { -6.0,  0.0}, { -6.0, -6.0}, {  0.0, -6.0},
        {  6.0, -6.0}, {  6.0,  6.0}, {  3.0,  6.0}, { -4.0, -3.0},
        {  4.0, -3.0}, {  7.0,  7.0}, { -7.0,  7.0}, { -7.0, -7.0},
        {  7.0, -7.0}, {  0.0,  8.0}, {  0.0, -8.0}, {  8.0,  0.0},
        { -8.0,  0.0}, { -5.0,  5.0}, {  5.0, -5.0},
    };
    size_t wp_idx  = 0;
    int    wp_hold = 0;
    double wp_x = waypoints[0].first;
    double wp_y = waypoints[0].second;

    // ── Initialise solver ────────────────────────────────────────────────────
    slam::GridSlam gs(num_particles, resolution, grid_half);
    gs.initialize(0.0, 0.0, 0.0);
    gs.set_motion_noise(0.05, 0.02);
    gs.set_sensor_model(0.65, -0.40, -5.0, 5.0);
    gs.set_scan_match_window(0.3, 0.15, 7, 7);

    // ── True pose ────────────────────────────────────────────────────────────
    double x_true = 0.0, y_true = 0.0, theta_true = 0.0;

    std::filesystem::create_directories("output");

    std::ofstream traj_file("output/grid_slam_trajectory.csv");
    traj_file << "step,t,x_true,y_true,theta_true,x_est,y_est,theta_est\n";

    std::ofstream part_file("output/grid_slam_particles.csv");
    part_file << "step,t,p_idx,x,y,theta,weight\n";

    // Incremental map snapshots: step,col,row,world_x,world_y,log_odds,prob
    std::ofstream snap_file("output/grid_slam_map_snapshots.csv");
    snap_file << "step,col,row,world_x,world_y,log_odds,prob\n";

    for (int step = 0; step < steps; ++step) {
        const double t = step * dt;

        // ── Choose next waypoint ──────────────────────────────────────────
        double dist_to_wp = std::hypot(wp_x - x_true, wp_y - y_true);
        bool   wp_reached = dist_to_wp < 1.0;
        bool   wp_stale   = wp_hold > 70;
        bool   wp_blocked = !line_of_sight(x_true, y_true, wp_x, wp_y, room);

        if (wp_reached || wp_stale || wp_blocked) {
            // Advance through the waypoint list, skip any that are blocked
            for (size_t tries = 0; tries < waypoints.size(); ++tries) {
                wp_idx = (wp_idx + 1) % waypoints.size();
                double cx = waypoints[wp_idx].first;
                double cy = waypoints[wp_idx].second;
                if (line_of_sight(x_true, y_true, cx, cy, room)) {
                    wp_x = cx; wp_y = cy;
                    break;
                }
            }
            wp_hold = 0;
        }
        ++wp_hold;

        // ── P-controller toward waypoint ──────────────────────────────────
        double desired_theta = std::atan2(wp_y - y_true, wp_x - x_true);
        double heading_err   = desired_theta - theta_true;
        while (heading_err >  M_PI) heading_err -= 2.0 * M_PI;
        while (heading_err < -M_PI) heading_err += 2.0 * M_PI;

        double w_cmd = std::max(-0.6, std::min(0.6, 2.5 * heading_err));
        double v_cmd = v_base * (1.0 - 0.5 * std::fabs(w_cmd) / 0.6);

        // ── Boundary safety: steer back toward centre ─────────────────────
        if (std::fabs(x_true) > room_bound || std::fabs(y_true) > room_bound) {
            double away = std::atan2(-y_true, -x_true);
            heading_err = away - theta_true;
            while (heading_err >  M_PI) heading_err -= 2.0 * M_PI;
            while (heading_err < -M_PI) heading_err += 2.0 * M_PI;
            w_cmd = std::max(-0.6, std::min(0.6, 3.5 * heading_err));
            v_cmd = v_base * 0.4;
            // Force next waypoint to be inside room
            wp_x = 0.0; wp_y = 0.0; wp_hold = 70;
        }

        // ── True motion ───────────────────────────────────────────────────
        x_true     += v_cmd * dt * std::cos(theta_true);
        y_true     += v_cmd * dt * std::sin(theta_true);
        theta_true += w_cmd * dt;
        while (theta_true >  M_PI) theta_true -= 2.0 * M_PI;
        while (theta_true < -M_PI) theta_true += 2.0 * M_PI;

        // ── Simulate laser scan ───────────────────────────────────────────
        slam::LaserScan scan = simulate_scan(x_true, y_true, theta_true,
                                              room, num_beams,
                                              range_max, rng, 0.05);

        // ── RBPF step ─────────────────────────────────────────────────────
        gs.process_scan(v_cmd, w_cmd, dt, scan);

        // ── Log trajectory ────────────────────────────────────────────────
        double xe, ye, te;
        gs.best_pose(xe, ye, te);
        traj_file << step << ',' << t << ','
                  << x_true << ',' << y_true << ',' << theta_true << ','
                  << xe     << ',' << ye     << ',' << te         << '\n';

        // ── Log particles ─────────────────────────────────────────────────
        const auto& parts = gs.particles();
        for (size_t pi = 0; pi < parts.size(); ++pi) {
            const auto& p = parts[pi];
            part_file << step << ',' << t << ',' << pi << ','
                      << p.x << ',' << p.y << ',' << p.theta << ','
                      << p.weight << '\n';
        }

        // ── Periodic map snapshot (best-particle) ─────────────────────────
        if (step % snap_every == 0) {
            const size_t bidx = gs.best_particle_index();
            const slam::OccupancyGrid& bg = gs.particles()[bidx].grid;
            for (int r = 0; r < bg.height; ++r) {
                for (int c = 0; c < bg.width; ++c) {
                    double lo = bg.at(c, r);
                    if (lo == 0.0) continue;
                    double wx, wy;
                    bg.cell_to_world(c, r, wx, wy);
                    double prob = slam::OccupancyGrid::log_odds_to_prob(lo);
                    snap_file << step << ',' << c << ',' << r << ','
                              << wx << ',' << wy << ','
                              << lo << ',' << prob << '\n';
                }
            }
        }
    }

    // ── Dump final best-particle SLAM map ─────────────────────────────────────
    std::ofstream map_file("output/grid_slam_map.csv");
    map_file << "col,row,world_x,world_y,log_odds,prob\n";

    const size_t best_idx = gs.best_particle_index();
    const slam::OccupancyGrid& best_grid = gs.particles()[best_idx].grid;

    for (int row = 0; row < best_grid.height; ++row) {
        for (int col = 0; col < best_grid.width; ++col) {
            double lo = best_grid.at(col, row);
            if (lo == 0.0) continue;
            double wx, wy;
            best_grid.cell_to_world(col, row, wx, wy);
            double prob = slam::OccupancyGrid::log_odds_to_prob(lo);
            map_file << col << ',' << row << ','
                     << wx  << ',' << wy  << ','
                     << lo  << ',' << prob << '\n';
        }
    }

    // ── Dump ground-truth map ─────────────────────────────────────────────────
    std::ofstream gt_file("output/grid_slam_gt_map.csv");
    gt_file << "col,row,world_x,world_y,log_odds,prob\n";

    slam::OccupancyGrid gt_grid = build_gt_map(room, resolution, grid_half);
    for (int row = 0; row < gt_grid.height; ++row) {
        for (int col = 0; col < gt_grid.width; ++col) {
            double lo = gt_grid.at(col, row);
            if (lo == 0.0) continue;
            double wx, wy;
            gt_grid.cell_to_world(col, row, wx, wy);
            double prob = slam::OccupancyGrid::log_odds_to_prob(lo);
            gt_file << col << ',' << row << ','
                    << wx  << ',' << wy  << ','
                    << lo  << ',' << prob << '\n';
        }
    }

    std::cout << "Wrote output/grid_slam_trajectory.csv, "
              << "output/grid_slam_particles.csv, "
              << "output/grid_slam_map.csv, "
              << "output/grid_slam_map_snapshots.csv, "
              << "output/grid_slam_gt_map.csv\n";
    return 0;
}
