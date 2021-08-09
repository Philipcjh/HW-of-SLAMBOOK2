cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


add_executable(pose_graph_Ceres_lie p3.cpp)
target_link_libraries(pose_graph_Ceres_lie
        ${CERES_LIBRARIES}
        ${CHOLMOD_LIBRARIES}
        ${Sophus_LIBRARIES}
        )