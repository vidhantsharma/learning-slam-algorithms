add_library(ekf_slam STATIC
    ekf_slam.cpp
    ekf_slam.h
)

target_include_directories(ekf_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ekf_slam PUBLIC utils)
