add_executable(ekf_slam_demo ekf_slam_demo.cpp)
target_link_libraries(ekf_slam_demo ekf_slam utils)

add_executable(fast_slam_demo fast_slam_demo.cpp)
target_link_libraries(fast_slam_demo fast_slam utils)

add_executable(grid_slam_demo grid_slam_demo.cpp)
target_link_libraries(grid_slam_demo grid_slam utils)
