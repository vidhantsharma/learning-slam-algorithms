add_library(pose_graph_slam STATIC
    pose_graph_slam.cpp
    pose_graph_slam.h
)

target_include_directories(pose_graph_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pose_graph_slam PUBLIC utils)
