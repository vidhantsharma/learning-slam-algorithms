#include "ekf_slam.h"

#include <cmath>
#include <stdexcept>

namespace slam {

EkfSlam::EkfSlam(size_t num_landmarks)
    : num_landmarks_(num_landmarks),
      state_(3 + 2 * num_landmarks, 1, 0.0),
      covariance_(kf::Matrix::identity(3 + 2 * num_landmarks) * 1e-3),
      process_noise_(kf::Matrix::identity(3 + 2 * num_landmarks) * 1e-3),
      measurement_noise_(kf::Matrix::identity(2) * 0.05),
      landmark_initialized_(num_landmarks, false) {}

void EkfSlam::initialize(double x, double y, double theta) {
    state_(0, 0) = x;
    state_(1, 0) = y;
    state_(2, 0) = theta;
    covariance_ = kf::Matrix::identity(state_.rows()) * 0.1;
    for (size_t i = 0; i < num_landmarks_; ++i) {
        landmark_initialized_[i] = false;
    }
}

void EkfSlam::set_motion_noise(double linear_std, double angular_std) {
    process_noise_ = kf::Matrix::identity(state_.rows()) * 1e-6;
    process_noise_(0, 0) = linear_std * linear_std;
    process_noise_(1, 1) = linear_std * linear_std;
    process_noise_(2, 2) = angular_std * angular_std;
}

void EkfSlam::set_measurement_noise(double range_std, double bearing_std) {
    measurement_noise_ = kf::Matrix::zeros(2, 2);
    measurement_noise_(0, 0) = range_std * range_std;
    measurement_noise_(1, 1) = bearing_std * bearing_std;
}

void EkfSlam::predict(double v, double w, double dt) {
    double theta = state_(2, 0);
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    state_(0, 0) += v * dt * cos_t;
    state_(1, 0) += v * dt * sin_t;
    state_(2, 0) = normalize_angle(theta + w * dt);

    kf::Matrix F = kf::Matrix::identity(state_.rows());
    F(0, 2) = -v * dt * sin_t;
    F(1, 2) = v * dt * cos_t;

    covariance_ = F * covariance_ * F.transpose() + process_noise_;
}

void EkfSlam::update(const std::vector<LandmarkObservation>& observations) {
    for (const auto& obs : observations) {
        if (obs.id < 0 || static_cast<size_t>(obs.id) >= num_landmarks_) {
            throw std::runtime_error("Landmark id out of range");
        }
        size_t lm_index = landmark_index(obs.id);

        if (!landmark_initialized_[obs.id]) {
            double theta = state_(2, 0);
            double lx = state_(0, 0) + obs.range * std::cos(theta + obs.bearing);
            double ly = state_(1, 0) + obs.range * std::sin(theta + obs.bearing);
            state_(lm_index, 0) = lx;
            state_(lm_index + 1, 0) = ly;
            landmark_initialized_[obs.id] = true;
        }

        double dx = state_(lm_index, 0) - state_(0, 0);
        double dy = state_(lm_index + 1, 0) - state_(1, 0);
        double q = dx * dx + dy * dy;
        double range = std::sqrt(q);
        double bearing = normalize_angle(std::atan2(dy, dx) - state_(2, 0));

        kf::Matrix z_pred(2, 1, 0.0);
        z_pred(0, 0) = range;
        z_pred(1, 0) = bearing;

        kf::Matrix H = kf::Matrix::zeros(2, state_.rows());
        if (range < 1e-6) {
            continue;
        }
        double inv_range = 1.0 / range;
        double inv_q = 1.0 / q;

        H(0, 0) = -dx * inv_range;
        H(0, 1) = -dy * inv_range;
        H(1, 0) = dy * inv_q;
        H(1, 1) = -dx * inv_q;
        H(1, 2) = -1.0;

        H(0, lm_index) = dx * inv_range;
        H(0, lm_index + 1) = dy * inv_range;
        H(1, lm_index) = -dy * inv_q;
        H(1, lm_index + 1) = dx * inv_q;

        kf::Matrix S = H * covariance_ * H.transpose() + measurement_noise_;
        kf::Matrix K = covariance_ * H.transpose() * S.inverse();

        kf::Matrix z(2, 1, 0.0);
        z(0, 0) = obs.range;
        z(1, 0) = obs.bearing;

        kf::Matrix y = z - z_pred;
        y(1, 0) = normalize_angle(y(1, 0));

        state_ = state_ + (K * y);
        kf::Matrix I = kf::Matrix::identity(state_.rows());
        covariance_ = (I - K * H) * covariance_;
    }
}

std::vector<double> EkfSlam::landmark_estimates() const {
    std::vector<double> estimates;
    estimates.reserve(num_landmarks_ * 2);
    for (size_t i = 0; i < num_landmarks_; ++i) {
        size_t idx = landmark_index(i);
        estimates.push_back(state_(idx, 0));
        estimates.push_back(state_(idx + 1, 0));
    }
    return estimates;
}

bool EkfSlam::is_landmark_initialized(size_t id) const {
    if (id >= landmark_initialized_.size()) {
        return false;
    }
    return landmark_initialized_[id];
}

size_t EkfSlam::landmark_index(size_t id) const {
    return 3 + 2 * id;
}

double EkfSlam::normalize_angle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

} // namespace slam
