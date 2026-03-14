add_library(grid_slam STATIC grid_slam.cpp grid_slam.h)
target_include_directories(grid_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grid_slam PUBLIC utils)
