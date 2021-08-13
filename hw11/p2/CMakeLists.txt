cmake_minimum_required(VERSION 2.8)
project(hw11)

set(CMAKE_CXX_FLAGS "-std=c++11")

# Eigen
include_directories("/usr/include/eigen3")

# opencv 
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(loop_closure_cal loop_closure_cal.cpp)
target_link_libraries(loop_closure_cal ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
