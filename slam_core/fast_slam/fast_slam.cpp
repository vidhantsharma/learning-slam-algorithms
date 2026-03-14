#include "fast_slam.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace slam {

// ─────────────────────────────────────────────────────────────────────────────
// Particle
// ─────────────────────────────────────────────────────────────────────────────

Particle::Particle(size_t num_landmarks)
    : landmarks(num_landmarks)
{
    for (auto& lm : landmarks) {
        lm.mean       = kf::Matrix(2, 1, 0.0);
        lm.covariance = kf::Matrix::identity(2) * 1000.0;
        lm.initialized = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FastSlam
// ─────────────────────────────────────────────────────────────────────────────

FastSlam::FastSlam(size_t num_particles, size_t num_landmarks)
    : num_particles_(num_particles),
      num_landmarks_(num_landmarks),
      particles_(num_particles, Particle(num_landmarks))
{}

void FastSlam::initialize(double x, double y, double theta) {
    for (auto& p : particles_) {
        p.x      = x;
        p.y      = y;
        p.theta  = theta;
        p.weight = 1.0 / static_cast<double>(num_particles_);
        for (auto& lm : p.landmarks) {
            lm.mean        = kf::Matrix(2, 1, 0.0);
            lm.covariance  = kf::Matrix::identity(2) * 1000.0;
            lm.initialized = false;
        }
    }
}

void FastSlam::set_motion_noise(double linear_std, double angular_std) {
    linear_std_  = linear_std;
    angular_std_ = angular_std;
}

void FastSlam::set_measurement_noise(double range_std, double bearing_std) {
    range_var_   = range_std * range_std;
    bearing_var_ = bearing_std * bearing_std;
}

// ─────────────────────────────────────────────────────────────────────────────
// predict  – sample a new pose for each particle from the motion model
//
//   x'     = x + v*dt*cos(θ) + n_x
//   y'     = y + v*dt*sin(θ) + n_y
//   θ'     = θ + w*dt        + n_θ
//
// where n ~ N(0, σ²) independently.
// ─────────────────────────────────────────────────────────────────────────────
void FastSlam::predict(double v, double w, double dt) {
    std::normal_distribution<double> dist_lin(0.0, linear_std_);
    std::normal_distribution<double> dist_ang(0.0, angular_std_);

    for (auto& p : particles_) {
        double cos_t = std::cos(p.theta);
        double sin_t = std::sin(p.theta);
        p.x     += v * dt * cos_t + dist_lin(rng_);
        p.y     += v * dt * sin_t + dist_lin(rng_);
        p.theta  = normalize_angle(p.theta + w * dt + dist_ang(rng_));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// update  – EKF landmark update for each particle, then weight by likelihood
//
// For each particle and each observation:
//   1. If landmark is unseen: initialise its mean/covariance
//   2. Compute predicted measurement z_hat from particle pose + landmark mean
//   3. Innovation covariance Q = H·Σ·Hᵀ + R
//   4. Kalman gain           K = Σ·Hᵀ·Q⁻¹
//   5. Update landmark mean  μ' = μ + K·(z - z_hat)
//   6. Update covariance     Σ' = (I - K·H)·Σ
//   7. Particle weight      *= N(z; z_hat, Q)   [likelihood]
// ─────────────────────────────────────────────────────────────────────────────
void FastSlam::update(const std::vector<FastSlamObservation>& observations) {
    kf::Matrix R = kf::Matrix::zeros(2, 2);
    R(0, 0) = range_var_;
    R(1, 1) = bearing_var_;

    for (auto& p : particles_) {
        double log_w = 0.0;   // accumulate log-likelihood per particle

        for (const auto& obs : observations) {
            if (obs.id < 0 || static_cast<size_t>(obs.id) >= num_landmarks_) {
                throw std::runtime_error("FastSLAM: landmark id out of range");
            }
            LandmarkEKF& lm = p.landmarks[obs.id];

            // ── Initialise landmark if first observation ──────────────────
            if (!lm.initialized) {
                lm.mean(0, 0) = p.x + obs.range * std::cos(p.theta + obs.bearing);
                lm.mean(1, 0) = p.y + obs.range * std::sin(p.theta + obs.bearing);
                // Covariance already set to 1000·I in constructor / initialize()
                lm.initialized = true;
                // No weight contribution for the first observation of a landmark
                continue;
            }

            // ── Predicted measurement ─────────────────────────────────────
            double dx = lm.mean(0, 0) - p.x;
            double dy = lm.mean(1, 0) - p.y;
            double q  = dx * dx + dy * dy;
            double r  = std::sqrt(q);

            if (r < 1e-6) {
                continue;
            }

            double z_hat_range   = r;
            double z_hat_bearing = normalize_angle(std::atan2(dy, dx) - p.theta);

            // ── Measurement Jacobian H (2×2, w.r.t. landmark position) ────
            kf::Matrix H = kf::Matrix::zeros(2, 2);
            H(0, 0) =  dx / r;
            H(0, 1) =  dy / r;
            H(1, 0) = -dy / q;
            H(1, 1) =  dx / q;

            // ── Innovation covariance Q = H·Σ·Hᵀ + R ─────────────────────
            kf::Matrix S = H * lm.covariance * H.transpose() + R;

            // ── Kalman gain K = Σ·Hᵀ·Q⁻¹ ─────────────────────────────────
            kf::Matrix K = lm.covariance * H.transpose() * S.inverse();

            // ── Innovation ────────────────────────────────────────────────
            kf::Matrix innov(2, 1, 0.0);
            innov(0, 0) = obs.range   - z_hat_range;
            innov(1, 0) = normalize_angle(obs.bearing - z_hat_bearing);

            // ── Update landmark mean and covariance ───────────────────────
            lm.mean       = lm.mean + K * innov;
            kf::Matrix I2 = kf::Matrix::identity(2);
            lm.covariance = (I2 - K * H) * lm.covariance;

            // ── Log-likelihood: log N(0; 0, S) contribution ───────────────
            // log p(z|x,m) = -0.5 * ( log(2π |S|) + νᵀ S⁻¹ ν )
            double det_S = S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0);
            if (det_S > 1e-12) {
                kf::Matrix S_inv = S.inverse();
                kf::Matrix innov_t = innov.transpose();
                kf::Matrix maha = innov_t * S_inv * innov;
                log_w += -0.5 * (std::log(2.0 * M_PI * det_S) + maha(0, 0));
            }
        }

        // Multiply particle weight by the observation likelihood
        p.weight *= std::exp(log_w);
    }

    // Normalise weights
    double total = 0.0;
    for (const auto& p : particles_) {
        total += p.weight;
    }
    if (total < 1e-300) {
        // Weight collapse – reset to uniform
        double uniform = 1.0 / static_cast<double>(num_particles_);
        for (auto& p : particles_) {
            p.weight = uniform;
        }
    } else {
        for (auto& p : particles_) {
            p.weight /= total;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// resample  – low-variance (systematic) resampling
// ─────────────────────────────────────────────────────────────────────────────
void FastSlam::resample() {
    const double N = static_cast<double>(num_particles_);

    // Build cumulative weight
    std::vector<double> cumulative(num_particles_);
    cumulative[0] = particles_[0].weight;
    for (size_t i = 1; i < num_particles_; ++i) {
        cumulative[i] = cumulative[i - 1] + particles_[i].weight;
    }

    // Low-variance resampling
    std::uniform_real_distribution<double> uniform(0.0, 1.0 / N);
    double u = uniform(rng_);

    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    size_t j = 0;
    for (size_t i = 0; i < num_particles_; ++i) {
        double threshold = u + static_cast<double>(i) / N;
        while (j < num_particles_ - 1 && cumulative[j] < threshold) {
            ++j;
        }
        new_particles.push_back(particles_[j]);
        new_particles.back().weight = 1.0 / N;
    }

    particles_ = std::move(new_particles);
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────

void FastSlam::best_pose(double& x, double& y, double& theta) const {
    size_t best = 0;
    for (size_t i = 1; i < num_particles_; ++i) {
        if (particles_[i].weight > particles_[best].weight) {
            best = i;
        }
    }
    x     = particles_[best].x;
    y     = particles_[best].y;
    theta = particles_[best].theta;
}

std::vector<double> FastSlam::mean_landmark_estimates() const {
    std::vector<double> estimates(num_landmarks_ * 2, 0.0);
    for (const auto& p : particles_) {
        for (size_t i = 0; i < num_landmarks_; ++i) {
            estimates[2 * i    ] += p.weight * p.landmarks[i].mean(0, 0);
            estimates[2 * i + 1] += p.weight * p.landmarks[i].mean(1, 0);
        }
    }
    return estimates;
}

std::vector<double> FastSlam::best_landmark_estimates() const {
    // Find best particle
    size_t best = 0;
    for (size_t i = 1; i < num_particles_; ++i) {
        if (particles_[i].weight > particles_[best].weight) {
            best = i;
        }
    }
    const Particle& bp = particles_[best];

    // Pack: id, x, y, cov_xx, cov_xy, cov_yy  (6 doubles per landmark)
    std::vector<double> out;
    out.reserve(num_landmarks_ * 6);
    for (size_t i = 0; i < num_landmarks_; ++i) {
        const LandmarkEKF& lm = bp.landmarks[i];
        out.push_back(static_cast<double>(i));
        out.push_back(lm.mean(0, 0));
        out.push_back(lm.mean(1, 0));
        out.push_back(lm.covariance(0, 0));
        out.push_back(lm.covariance(0, 1));
        out.push_back(lm.covariance(1, 1));
    }
    return out;
}

double FastSlam::normalize_angle(double angle) {
    while (angle >  M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

} // namespace slam
