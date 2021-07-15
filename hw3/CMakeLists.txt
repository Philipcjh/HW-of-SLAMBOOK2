cmake_minimum_required(VERSION 2.8)
project(hw3)

set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories("/usr/include/eigen3")
add_executable(matrix p5.cpp)
add_executable(solve p6.cpp)



