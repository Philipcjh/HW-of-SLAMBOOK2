cmake_minimum_required(VERSION 2.8)
project(hw7)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(features p2.cpp)
target_link_libraries(features ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d p6.cpp)
target_link_libraries(pose_estimation_3d2d
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

add_executable(pose_estimation_3d3d p7.cpp)
target_link_libraries(pose_estimation_3d3d
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

add_executable(ceresPnP p10PnP.cpp)
target_link_libraries(ceresPnP ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable(ceresICP p10ICP.cpp)
target_link_libraries(ceresICP ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)