add_library(fast_slam STATIC fast_slam.cpp fast_slam.h)
target_include_directories(fast_slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fast_slam PUBLIC utils)
