cmake_minimum_required(VERSION 3.15)
project(nncpp_exec)

set(CMAKE_CXX_STANDARD 20)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -lblas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

add_executable(nncpp_exec main.cpp Matrix.cpp Matrix.h Layer.cpp Layer.h NeuralNet.cpp NeuralNet.h DataSetOrganizer.cpp DataSetOrganizer.h)