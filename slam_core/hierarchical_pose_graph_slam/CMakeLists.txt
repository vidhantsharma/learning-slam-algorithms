add_library(hierarchical_pose_graph_slam STATIC
    hierarchical_pose_graph_slam.cpp
    hierarchical_pose_graph_slam.h
)

target_include_directories(hierarchical_pose_graph_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hierarchical_pose_graph_slam PUBLIC pose_graph_slam utils)
