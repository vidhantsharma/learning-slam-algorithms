#include "grid_slam.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// OccupancyGrid helpers
// ─────────────────────────────────────────────────────────────────────────────

void OccupancyGrid::world_to_cell(double wx, double wy, int& col, int& row) const {
    col = static_cast<int>(std::floor((wx - origin_x) / resolution + 0.5));
    row = static_cast<int>(std::floor((wy - origin_y) / resolution + 0.5));
}

void OccupancyGrid::cell_to_world(int col, int row, double& wx, double& wy) const {
    wx = origin_x + col * resolution;
    wy = origin_y + row * resolution;
}

bool OccupancyGrid::in_bounds(int col, int row) const {
    return col >= 0 && col < width && row >= 0 && row < height;
}

double OccupancyGrid::log_odds_to_prob(double lo) {
    return 1.0 - 1.0 / (1.0 + std::exp(lo));
}

// ─────────────────────────────────────────────────────────────────────────────
// GridSlam constructor / configuration
// ─────────────────────────────────────────────────────────────────────────────

GridSlam::GridSlam(size_t num_particles, double resolution, double grid_half_size)
    : num_particles_(num_particles),
      resolution_(resolution),
      grid_half_size_(grid_half_size),
      particles_(num_particles)
{
    for (auto& p : particles_) {
        p.grid = make_empty_grid();
    }
}

void GridSlam::initialize(double x, double y, double theta) {
    for (auto& p : particles_) {
        p.x      = x;
        p.y      = y;
        p.theta  = theta;
        p.weight = 1.0 / static_cast<double>(num_particles_);
        p.grid   = make_empty_grid();
    }
}

void GridSlam::set_motion_noise(double linear_std, double angular_std) {
    linear_std_  = linear_std;
    angular_std_ = angular_std;
}

void GridSlam::set_sensor_model(double lo_occ, double lo_free,
                                double lo_min,  double lo_max) {
    lo_occ_  = lo_occ;
    lo_free_ = lo_free;
    lo_min_  = lo_min;
    lo_max_  = lo_max;
}

void GridSlam::set_scan_match_window(double xy_range, double theta_range,
                                     int xy_steps,    int theta_steps) {
    sm_xy_range_    = xy_range;
    sm_theta_range_ = theta_range;
    sm_xy_steps_    = (xy_steps    | 1);   // force odd so centre is sampled
    sm_theta_steps_ = (theta_steps | 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// make_empty_grid
// ─────────────────────────────────────────────────────────────────────────────

OccupancyGrid GridSlam::make_empty_grid() const {
    OccupancyGrid g;
    g.resolution = resolution_;
    int cells    = static_cast<int>(std::ceil(2.0 * grid_half_size_ / resolution_));
    g.width      = cells;
    g.height     = cells;
    g.origin_x   = -grid_half_size_;
    g.origin_y   = -grid_half_size_;
    g.log_odds.assign(static_cast<size_t>(cells * cells), 0.0);
    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// sample_pose – draw a new pose from the motion model (velocity model)
// ─────────────────────────────────────────────────────────────────────────────

void GridSlam::sample_pose(double xi, double yi, double ti,
                            double v,  double w,  double dt,
                            double& xo, double& yo, double& to) {
    std::normal_distribution<double> nd_lin(0.0, linear_std_);
    std::normal_distribution<double> nd_ang(0.0, angular_std_);
    xo = xi + v * dt * std::cos(ti) + nd_lin(rng_);
    yo = yi + v * dt * std::sin(ti) + nd_lin(rng_);
    to = normalize_angle(ti + w * dt + nd_ang(rng_));
}

// ─────────────────────────────────────────────────────────────────────────────
// scan_score – score how well the scan endpoints fit the current grid map.
//
// Returns the sum of log-odds values at all valid scan endpoint cells.
// Higher is better (more cells are already marked occupied).
// ─────────────────────────────────────────────────────────────────────────────

double GridSlam::scan_score(const OccupancyGrid& grid,
                             double x, double y, double theta,
                             const LaserScan& scan) const {
    const size_t n     = scan.ranges.size();
    const double step  = (n > 1) ? (scan.angle_max - scan.angle_min)
                                   / static_cast<double>(n - 1)
                                 : 0.0;
    double score = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = scan.ranges[i];
        if (!std::isfinite(r) || r >= scan.range_max) continue;

        double angle = theta + scan.angle_min + i * step;
        double wx    = x + r * std::cos(angle);
        double wy    = y + r * std::sin(angle);

        int col, row;
        grid.world_to_cell(wx, wy, col, row);
        if (grid.in_bounds(col, row)) {
            score += grid.at(col, row);   // log-odds: positive = occupied
        }
    }
    return score;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bresenham ray-update
// ─────────────────────────────────────────────────────────────────────────────

void GridSlam::bresenham_update(OccupancyGrid& grid,
                                 int col0, int row0,
                                 int col1, int row1,
                                 bool hit) {
    // Standard Bresenham line traversal
    int dx = std::abs(col1 - col0);
    int dy = std::abs(row1 - row0);
    int sx = (col0 < col1) ? 1 : -1;
    int sy = (row0 < row1) ? 1 : -1;
    int err = dx - dy;

    int col = col0, row = row0;
    while (true) {
        bool is_endpoint = (col == col1 && row == row1);

        if (grid.in_bounds(col, row)) {
            if (is_endpoint && hit) {
                // Mark occupied
                grid.at(col, row) = clamp(grid.at(col, row) + lo_occ_,
                                          lo_min_, lo_max_);
            } else if (!is_endpoint) {
                // Mark free (cells along the ray before the endpoint)
                grid.at(col, row) = clamp(grid.at(col, row) + lo_free_,
                                          lo_min_, lo_max_);
            }
        }

        if (is_endpoint) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; col += sx; }
        if (e2 <  dx) { err += dx; row += sy; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate_scan – update grid with one laser scan at pose (x, y, theta)
// ─────────────────────────────────────────────────────────────────────────────

void GridSlam::integrate_scan(OccupancyGrid& grid,
                               double x, double y, double theta,
                               const LaserScan& scan) {
    int col0, row0;
    grid.world_to_cell(x, y, col0, row0);

    const size_t n    = scan.ranges.size();
    const double step = (n > 1) ? (scan.angle_max - scan.angle_min)
                                  / static_cast<double>(n - 1)
                                : 0.0;

    for (size_t i = 0; i < n; ++i) {
        double r = scan.ranges[i];
        if (!std::isfinite(r)) continue;

        bool hit = (r < scan.range_max);
        if (!hit) r = scan.range_max;   // still trace the free space

        double angle = theta + scan.angle_min + i * step;
        double wx    = x + r * std::cos(angle);
        double wy    = y + r * std::sin(angle);

        int col1, row1;
        grid.world_to_cell(wx, wy, col1, row1);
        bresenham_update(grid, col0, row0, col1, row1, hit);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// process_scan  – the main RBPF step
//
// For each particle:
//   1. Sample a new pose from the motion model (predict)
//   2. Scan-match: search a small window around the predicted pose to find
//      the pose that best explains the scan given the particle's current map.
//      This is the "improved proposal" from FastSLAM 2.0 / GMapping.
//   3. Compute importance weight as the scan-likelihood at the best pose.
//   4. Accumulate weight, normalise, resample, then integrate the scan.
// ─────────────────────────────────────────────────────────────────────────────

void GridSlam::process_scan(double v, double w, double dt, const LaserScan& scan) {
    // ── Step 1 & 2: predict + scan-match per particle ────────────────────────
    for (auto& p : particles_) {
        // Sample a predicted pose
        double xp, yp, tp;
        sample_pose(p.x, p.y, p.theta, v, w, dt, xp, yp, tp);

        // Skip scan-matching if the map is empty (first scan)
        bool map_has_data = false;
        for (double lo : p.grid.log_odds) {
            if (lo != 0.0) { map_has_data = true; break; }
        }

        if (!map_has_data) {
            // First scan: accept predicted pose, weight stays uniform
            p.x = xp; p.y = yp; p.theta = tp;
        } else {
            // Scan-match: search a window around the predicted pose
            double best_score = -std::numeric_limits<double>::infinity();
            double best_x = xp, best_y = yp, best_t = tp;

            int hx = sm_xy_steps_    / 2;
            int ht = sm_theta_steps_ / 2;
            double dxy = (sm_xy_steps_    > 1) ? sm_xy_range_    / hx : 0.0;
            double dt2 = (sm_theta_steps_ > 1) ? sm_theta_range_ / ht : 0.0;

            for (int ti = -ht; ti <= ht; ++ti) {
                for (int xi = -hx; xi <= hx; ++xi) {
                    for (int yi = -hx; yi <= hx; ++yi) {
                        double cx = xp + xi * dxy;
                        double cy = yp + yi * dxy;
                        double ct = normalize_angle(tp + ti * dt2);
                        double s  = scan_score(p.grid, cx, cy, ct, scan);
                        if (s > best_score) {
                            best_score = s;
                            best_x = cx; best_y = cy; best_t = ct;
                        }
                    }
                }
            }

            p.x = best_x; p.y = best_y; p.theta = best_t;

            // ── Step 3: importance weight from scan likelihood ───────────────
            // best_score already holds scan_score() at the best pose; no need
            // to call scan_score() again.

            // Count valid beams for normalisation
            size_t valid = 0;
            for (double r : scan.ranges)
                if (std::isfinite(r) && r < scan.range_max) ++valid;
            if (valid == 0) valid = 1;

            p.weight = std::exp(best_score / static_cast<double>(valid));
        }
    }

    // ── Normalise weights ────────────────────────────────────────────────────
    double total = 0.0;
    for (const auto& p : particles_) total += p.weight;

    if (total < 1e-300) {
        double uni = 1.0 / static_cast<double>(num_particles_);
        for (auto& p : particles_) p.weight = uni;
    } else {
        for (auto& p : particles_) p.weight /= total;
    }

    // ── Step 4: integrate scan into each particle's map ──────────────────────
    for (auto& p : particles_) {
        integrate_scan(p.grid, p.x, p.y, p.theta, scan);
    }

    // ── Step 5: resample ─────────────────────────────────────────────────────
    resample();
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-variance resampling
// ─────────────────────────────────────────────────────────────────────────────

void GridSlam::resample() {
    const double N = static_cast<double>(num_particles_);

    std::vector<double> cumulative(num_particles_);
    cumulative[0] = particles_[0].weight;
    for (size_t i = 1; i < num_particles_; ++i)
        cumulative[i] = cumulative[i - 1] + particles_[i].weight;

    std::uniform_real_distribution<double> uni(0.0, 1.0 / N);
    double u = uni(rng_);

    std::vector<GridParticle> new_particles;
    new_particles.reserve(num_particles_);

    size_t j = 0;
    for (size_t i = 0; i < num_particles_; ++i) {
        double threshold = u + static_cast<double>(i) / N;
        while (j < num_particles_ - 1 && cumulative[j] < threshold) ++j;
        new_particles.push_back(particles_[j]);
        new_particles.back().weight = 1.0 / N;
    }
    particles_ = std::move(new_particles);
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────

size_t GridSlam::best_particle_index() const {
    size_t best = 0;
    for (size_t i = 1; i < num_particles_; ++i)
        if (particles_[i].weight > particles_[best].weight) best = i;
    return best;
}

void GridSlam::best_pose(double& x, double& y, double& theta) const {
    const auto& p = particles_[best_particle_index()];
    x = p.x; y = p.y; theta = p.theta;
}

void GridSlam::mean_pose(double& x, double& y, double& theta) const {
    x = 0.0; y = 0.0; theta = 0.0;
    for (const auto& p : particles_) {
        x     += p.weight * p.x;
        y     += p.weight * p.y;
        theta += p.weight * p.theta;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

double GridSlam::normalize_angle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

double GridSlam::clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

} // namespace slam
