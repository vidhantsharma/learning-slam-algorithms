#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Laser scan: one 360-degree sweep stored as N range readings, uniformly
// spaced in angle from angle_min to angle_max.
// ─────────────────────────────────────────────────────────────────────────────
struct LaserScan {
    std::vector<double> ranges;   // metres, NaN/inf means no return
    double angle_min{-M_PI};      // first beam angle (rad, robot frame)
    double angle_max{ M_PI};      // last  beam angle (rad, robot frame)
    double range_max{10.0};       // max usable range (metres)
};

// ─────────────────────────────────────────────────────────────────────────────
// OccupancyGrid – 2-D log-odds grid
//
// Cells store log-odds values.  Positive → occupied, negative → free.
// The grid is axis-aligned in world coordinates.
// ─────────────────────────────────────────────────────────────────────────────
struct OccupancyGrid {
    int    width{0};          // number of cells in X
    int    height{0};         // number of cells in Y
    double resolution{0.1};   // metres per cell
    double origin_x{0.0};     // world X of cell (0,0) centre
    double origin_y{0.0};     // world Y of cell (0,0) centre

    std::vector<double> log_odds;  // row-major: [row * width + col]

    // Convenience helpers
    double& at(int col, int row)       { return log_odds[row * width + col]; }
    double  at(int col, int row) const { return log_odds[row * width + col]; }

    // World ↔ cell conversions
    void world_to_cell(double wx, double wy, int& col, int& row) const;
    void cell_to_world(int col, int row, double& wx, double& wy) const;
    bool in_bounds(int col, int row) const;

    // Probability of occupancy from log-odds
    static double log_odds_to_prob(double lo);
};

// ─────────────────────────────────────────────────────────────────────────────
// One FastSLAM 2.0 (RBPF) particle
// ─────────────────────────────────────────────────────────────────────────────
struct GridParticle {
    double x{0.0};
    double y{0.0};
    double theta{0.0};
    double weight{1.0};
    OccupancyGrid grid;
};

// ─────────────────────────────────────────────────────────────────────────────
// GridSlam  –  Rao-Blackwellised Particle Filter (FastSLAM 2.0 style)
//              using a 2-D occupancy grid as the map representation.
//
// Algorithm per timestep:
//   1. predict   – sample new pose from motion model
//   2. scan_match – local scan-matching to improve the proposal distribution
//   3. weight    – evaluate importance weight from scan-to-map match
//   4. resample  – low-variance resampling
//   5. update    – integrate the scan into each particle's map
// ─────────────────────────────────────────────────────────────────────────────
class GridSlam {
public:
    // grid_half_size: half-width/height of the map in metres
    GridSlam(size_t num_particles,
             double resolution       = 0.1,
             double grid_half_size   = 25.0);

    void initialize(double x, double y, double theta);
    void set_motion_noise(double linear_std, double angular_std);

    // Sensor model log-odds increments
    void set_sensor_model(double log_odds_occ  =  0.65,
                          double log_odds_free = -0.40,
                          double log_odds_min  = -5.0,
                          double log_odds_max  =  5.0);

    // Scan-match window (metres / radians) used for proposal improvement
    void set_scan_match_window(double xy_range    = 0.3,
                               double theta_range = 0.15,
                               int    xy_steps    = 7,
                               int    theta_steps = 7);

    // Main RBPF step: predict + weight + resample + update
    void process_scan(double v, double w, double dt, const LaserScan& scan);

    // ── Accessors ────────────────────────────────────────────────────────────
    const std::vector<GridParticle>& particles() const { return particles_; }

    // Best-weight particle pose
    void best_pose(double& x, double& y, double& theta) const;

    // Index of best particle
    size_t best_particle_index() const;

    // Weighted-mean pose
    void mean_pose(double& x, double& y, double& theta) const;

private:
    // ── Parameters ───────────────────────────────────────────────────────────
    size_t num_particles_;
    double resolution_;
    double grid_half_size_;

    double linear_std_{0.05};
    double angular_std_{0.02};

    double lo_occ_{0.65};
    double lo_free_{-0.40};
    double lo_min_{-5.0};
    double lo_max_{ 5.0};

    double sm_xy_range_{0.3};
    double sm_theta_range_{0.15};
    int    sm_xy_steps_{7};
    int    sm_theta_steps_{7};

    std::vector<GridParticle> particles_;
    std::default_random_engine rng_{42};

    // ── Internal helpers ──────────────────────────────────────────────────────
    OccupancyGrid make_empty_grid() const;

    // Draw a new pose from motion model around the given pose
    void sample_pose(double x_in, double y_in, double theta_in,
                     double v, double w, double dt,
                     double& x_out, double& y_out, double& theta_out);

    // Score: sum of log-odds of scan endpoints in the given map at pose (x,y,θ)
    double scan_score(const OccupancyGrid& grid,
                      double x, double y, double theta,
                      const LaserScan& scan) const;

    // Integrate one scan into the grid (inverse sensor model via Bresenham)
    void integrate_scan(OccupancyGrid& grid,
                        double x, double y, double theta,
                        const LaserScan& scan);

    // Ray-trace from (x0,y0) to (x1,y1) and mark cells free / occupied
    void bresenham_update(OccupancyGrid& grid,
                          int col0, int row0,
                          int col1, int row1,
                          bool hit);

    // Low-variance resampling (same as FastSLAM 1.0)
    void resample();

    static double normalize_angle(double a);
    static double clamp(double v, double lo, double hi);
};

} // namespace slam
