#pragma once

#include <vector>
#include <cstddef>
#include <random>

#include "../utils/matrix.h"

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct FastSlamObservation {
    int    id;      // known data-association (ground-truth landmark id)
    double range;
    double bearing;
};

// Per-landmark EKF stored inside each particle.
struct LandmarkEKF {
    kf::Matrix mean;        // 2×1
    kf::Matrix covariance;  // 2×2
    bool       initialized{false};
};

// One particle: a pose hypothesis + one EKF per landmark.
struct Particle {
    double x{0.0};
    double y{0.0};
    double theta{0.0};
    double weight{1.0};
    std::vector<LandmarkEKF> landmarks;     // size == num_landmarks

    explicit Particle(size_t num_landmarks);
};

// ─────────────────────────────────────────────────────────────────────────────
// FastSLAM 1.0
//
// Algorithm (per timestep):
//   1. predict  – draw a new pose for every particle from the motion model
//   2. update   – for each particle, run a per-landmark EKF update and
//                 accumulate the observation weight
//   3. resample – low-variance resampling to focus on high-weight particles
// ─────────────────────────────────────────────────────────────────────────────
class FastSlam {
public:
    FastSlam(size_t num_particles, size_t num_landmarks);

    void initialize(double x, double y, double theta);
    void set_motion_noise(double linear_std, double angular_std);
    void set_measurement_noise(double range_std, double bearing_std);

    void predict(double v, double w, double dt);
    void update(const std::vector<FastSlamObservation>& observations);
    void resample();

    // ── Accessors ────────────────────────────────────────────────────────────
    const std::vector<Particle>& particles() const { return particles_; }

    // Best (highest-weight) particle pose
    void best_pose(double& x, double& y, double& theta) const;

    // Weighted-mean landmark positions for all particles (for logging).
    // Returns {x0, y0, x1, y1, …} for each landmark.
    std::vector<double> mean_landmark_estimates() const;

    // Best-particle landmark estimates: {id, x, y, cov_xx, cov_xy, cov_yy, …}
    std::vector<double> best_landmark_estimates() const;

private:
    size_t num_particles_;
    size_t num_landmarks_;

    std::vector<Particle> particles_;

    double linear_std_{0.05};
    double angular_std_{0.02};
    double range_var_{0.15 * 0.15};
    double bearing_var_{0.05 * 0.05};

    std::default_random_engine rng_{42};

    static double normalize_angle(double angle);
};

} // namespace slam
