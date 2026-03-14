#include "../slam_core/fast_slam/fast_slam.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Landmark {
    int    id;
    double x;
    double y;
};

static std::vector<Landmark> create_landmarks() {
    return {
        {0,  5.0,  5.0},
        {1, -4.0,  6.0},
        {2,  8.0, -3.0},
        {3, -6.0, -5.0},
        {4,  0.0, -7.0}
    };
}

static int parse_steps(int argc, char** argv) {
    int steps = 250;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[i + 1]);
            ++i;
        }
    }
    return steps;
}

int main(int argc, char** argv) {
    const std::vector<Landmark> landmarks = create_landmarks();
    const size_t num_landmarks = landmarks.size();
    const size_t num_particles = 50;

    slam::FastSlam fs(num_particles, num_landmarks);
    fs.initialize(0.0, 0.0, 0.0);
    fs.set_motion_noise(0.05, 0.02);
    fs.set_measurement_noise(0.15, 0.05);

    // Measurement noise for the simulated sensor
    std::default_random_engine rng(42);
    std::normal_distribution<double> range_noise(0.0, 0.15);
    std::normal_distribution<double> bearing_noise(0.0, 0.05);

    double x_true     = 0.0;
    double y_true     = 0.0;
    double theta_true = 0.0;

    const double v        = 0.8;
    const double w        = 0.12;
    const double dt       = 0.1;
    const double max_range = 10.0;

    const int steps = parse_steps(argc, argv);

    std::filesystem::create_directories("output");

    // ── Trajectory CSV ─────────────────────────────────────────────────────
    std::ofstream traj_file("output/fast_slam_trajectory.csv");
    traj_file << "step,t,x_true,y_true,theta_true,x_est,y_est,theta_est\n";

    // ── Landmark estimates per step (best particle) ────────────────────────
    std::ofstream lm_time_file("output/fast_slam_landmark_estimates.csv");
    lm_time_file << "step,t,id,x_est,y_est,cov_xx,cov_xy,cov_yy\n";

    // ── Particle cloud per step ────────────────────────────────────────────
    std::ofstream particle_file("output/fast_slam_particles.csv");
    particle_file << "step,t,p_idx,x,y,theta,weight\n";

    for (int step = 0; step < steps; ++step) {
        const double t = step * dt;

        // ── True motion ───────────────────────────────────────────────────
        x_true     += v * dt * std::cos(theta_true);
        y_true     += v * dt * std::sin(theta_true);
        theta_true += w * dt;

        // ── FastSLAM predict ──────────────────────────────────────────────
        fs.predict(v, w, dt);

        // ── Simulate sensor ───────────────────────────────────────────────
        std::vector<slam::FastSlamObservation> observations;
        for (const auto& lm : landmarks) {
            double dx    = lm.x - x_true;
            double dy    = lm.y - y_true;
            double range = std::sqrt(dx * dx + dy * dy);
            if (range > max_range) {
                continue;
            }
            double bearing = std::atan2(dy, dx) - theta_true;
            observations.push_back({lm.id,
                                     range + range_noise(rng),
                                     bearing + bearing_noise(rng)});
        }

        // ── FastSLAM update + resample ────────────────────────────────────
        fs.update(observations);
        fs.resample();

        // ── Log best-particle pose ────────────────────────────────────────
        double x_est, y_est, theta_est;
        fs.best_pose(x_est, y_est, theta_est);
        traj_file << step << ',' << t << ','
                  << x_true << ',' << y_true << ',' << theta_true << ','
                  << x_est  << ',' << y_est  << ',' << theta_est  << '\n';

        // ── Log best-particle landmark estimates ──────────────────────────
        const std::vector<double> lm_est = fs.best_landmark_estimates();
        for (size_t i = 0; i < num_landmarks; ++i) {
            size_t off = i * 6;
            int    id      = static_cast<int>(lm_est[off]);
            double x_l     = lm_est[off + 1];
            double y_l     = lm_est[off + 2];
            double cov_xx  = lm_est[off + 3];
            double cov_xy  = lm_est[off + 4];
            double cov_yy  = lm_est[off + 5];
            lm_time_file << step << ',' << t << ',' << id << ','
                         << x_l << ',' << y_l << ','
                         << cov_xx << ',' << cov_xy << ',' << cov_yy << '\n';
        }

        // ── Log particle cloud (every step) ────────────────────────────────
        if (step % 1 == 0) {
            const auto& particles = fs.particles();
            for (size_t pi = 0; pi < particles.size(); ++pi) {
                const auto& p = particles[pi];
                particle_file << step << ',' << t << ',' << pi << ','
                              << p.x << ',' << p.y << ',' << p.theta << ','
                              << p.weight << '\n';
            }
        }
    }

    // ── Final landmark summary ────────────────────────────────────────────
    std::ofstream lm_file("output/fast_slam_landmarks.csv");
    lm_file << "id,x_true,y_true,x_est,y_est\n";

    const std::vector<double> final_lm = fs.mean_landmark_estimates();
    for (const auto& lm : landmarks) {
        size_t i = static_cast<size_t>(lm.id);
        lm_file << lm.id << ',' << lm.x << ',' << lm.y << ','
                << final_lm[2 * i] << ',' << final_lm[2 * i + 1] << '\n';
    }

    std::cout << "Wrote output/fast_slam_trajectory.csv, "
              << "output/fast_slam_landmarks.csv, "
              << "output/fast_slam_landmark_estimates.csv, "
              << "output/fast_slam_particles.csv\n";
    return 0;
}
