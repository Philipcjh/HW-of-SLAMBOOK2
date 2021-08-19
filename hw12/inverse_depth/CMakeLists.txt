cmake_minimum_required(VERSION 2.8)
project(inverse_depth)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES})

add_executable(inverse_depth_mapping inverse_depth_mapping.cpp)
target_link_libraries(inverse_depth_mapping ${THIRD_PARTY_LIBS})