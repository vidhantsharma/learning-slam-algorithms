#pragma once

#include <vector>
#include <cstddef>

#include "../utils/matrix.h"

namespace slam {

struct LandmarkObservation {
    int id;
    double range;
    double bearing;
};

class EkfSlam {
public:
    EkfSlam(size_t num_landmarks);

    void initialize(double x, double y, double theta);
    void set_motion_noise(double linear_std, double angular_std);
    void set_measurement_noise(double range_std, double bearing_std);

    void predict(double v, double w, double dt);
    void update(const std::vector<LandmarkObservation>& observations);

    const kf::Matrix& state() const { return state_; }
    const kf::Matrix& covariance() const { return covariance_; }
    bool is_landmark_initialized(size_t id) const;

    std::vector<double> landmark_estimates() const;

private:
    size_t num_landmarks_{};
    kf::Matrix state_;
    kf::Matrix covariance_;
    kf::Matrix process_noise_;
    kf::Matrix measurement_noise_;
    std::vector<bool> landmark_initialized_;

    size_t landmark_index(size_t id) const;
    static double normalize_angle(double angle);
};

} // namespace slam
