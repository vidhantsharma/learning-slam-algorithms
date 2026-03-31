add_library(robust_pose_graph_slam STATIC
    robust_pose_graph_slam.cpp
    robust_pose_graph_slam.h
)

target_include_directories(robust_pose_graph_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(robust_pose_graph_slam PUBLIC utils)
