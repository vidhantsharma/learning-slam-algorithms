cmake_minimum_required(VERSION 3.16)
project(slam_algorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(slam_core/utils)
add_subdirectory(slam_core/ekf_slam)
add_subdirectory(slam_core/fast_slam)
add_subdirectory(slam_core/grid_slam)
add_subdirectory(examples)
