#include "../slam_core/ekf_slam/ekf_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Landmark {
    int id;
    double x;
    double y;
};

static std::vector<Landmark> create_landmarks() {
    return {
        {0, 5.0, 5.0},
        {1, -4.0, 6.0},
        {2, 8.0, -3.0},
        {3, -6.0, -5.0},
        {4, 0.0, -7.0}
    };
}

static int parse_steps(int argc, char** argv) {
    int steps = 250;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[i + 1]);
            i++;
        }
    }
    return steps;
}

int main(int argc, char** argv) {
    std::vector<Landmark> landmarks = create_landmarks();
    const size_t num_landmarks = landmarks.size();

    slam::EkfSlam ekf(num_landmarks);
    ekf.initialize(0.0, 0.0, 0.0);
    ekf.set_motion_noise(0.05, 0.02);
    ekf.set_measurement_noise(0.15, 0.05);

    std::default_random_engine rng(42);
    std::normal_distribution<double> range_noise(0.0, 0.15);
    std::normal_distribution<double> bearing_noise(0.0, 0.05);

    double x_true = 0.0;
    double y_true = 0.0;
    double theta_true = 0.0;

    double v = 0.8;
    double w = 0.12;
    double dt = 0.1;
    int steps = parse_steps(argc, argv);
    double max_range = 10.0;

    std::filesystem::create_directories("output");
    std::ofstream traj_file("output/trajectory.csv");
    traj_file << "step,t,x_true,y_true,theta_true,x_est,y_est,theta_est\n";

    std::ofstream lm_time_file("output/landmark_estimates.csv");
    lm_time_file << "step,t,id,x_est,y_est,cov_xx,cov_xy,cov_yy\n";

    std::ofstream obs_file("output/observations.csv");
    obs_file << "step,t,obs_x,obs_y,assoc_id,range,bearing\n";

    for (int step = 0; step < steps; ++step) {
        double t = step * dt;

        x_true += v * dt * std::cos(theta_true);
        y_true += v * dt * std::sin(theta_true);
        theta_true += w * dt;

        ekf.predict(v, w, dt);

        std::vector<slam::LandmarkObservation> observations;
        for (const auto& lm : landmarks) {
            double dx = lm.x - x_true;
            double dy = lm.y - y_true;
            double range = std::sqrt(dx * dx + dy * dy);
            if (range > max_range) {
                continue;
            }
            double bearing = std::atan2(dy, dx) - theta_true;
            observations.push_back({lm.id, range + range_noise(rng), bearing + bearing_noise(rng)});
        }

        ekf.update(observations);

        const auto& state = ekf.state();
        traj_file << step << ',' << t << ','
                  << x_true << ',' << y_true << ',' << theta_true << ','
                  << state(0, 0) << ',' << state(1, 0) << ',' << state(2, 0) << '\n';

        const auto& cov = ekf.covariance();
        for (const auto& lm : landmarks) {
            size_t idx = 3 + 2 * lm.id;
            double cov_xx = cov(idx, idx);
            double cov_xy = cov(idx, idx + 1);
            double cov_yy = cov(idx + 1, idx + 1);
            lm_time_file << step << ',' << t << ',' << lm.id << ','
                         << state(idx, 0) << ',' << state(idx + 1, 0) << ','
                         << cov_xx << ',' << cov_xy << ',' << cov_yy << '\n';
        }
    }

    std::ofstream lm_file("output/landmarks.csv");
    lm_file << "id,x_true,y_true,x_est,y_est\n";
    const auto& state = ekf.state();
    for (const auto& lm : landmarks) {
        size_t idx = 3 + 2 * lm.id;
        lm_file << lm.id << ',' << lm.x << ',' << lm.y << ','
                << state(idx, 0) << ',' << state(idx + 1, 0) << '\n';
    }

    std::cout << "Wrote output/trajectory.csv, output/landmarks.csv, output/landmark_estimates.csv, and output/observations.csv" << std::endl;
    return 0;
}
