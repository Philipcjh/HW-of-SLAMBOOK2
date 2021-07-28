cmake_minimum_required(VERSION 2.8)
project(ch8)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)


add_executable(direct_method p3.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(ceres_direct_method p4_Ceres_direct_method.cpp)
target_link_libraries(ceres_direct_method ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)

add_executable(g2o_direct_method p4_g2o_direct_method.cpp)
target_link_libraries(g2o_direct_method
        g2o_core g2o_stuff
        ${OpenCV_LIBS})