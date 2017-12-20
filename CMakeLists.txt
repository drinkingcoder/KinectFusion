cmake_minimum_required(VERSION 3.2)
project(KinectFusion)

set(EIGEN3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/Eigen)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(TOON_INCLUDE_PATHS ${PROJECT_SOURCE_DIR}/3rdparty)

find_package(CUDA REQUIRED)
#find_package(Pangolin 0.4 REQUIRED)
find_package(OpenGL)
find_package(GLUT)
find_package(OpenCV 2.4.9 REQUIRED)
message(STATUS "OpenCV version: " ${OpenCV_VERSION})
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 2.4.9 not found.")
endif()

set(CUDA_HOST_COMPILER "g++")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/3rdparty)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
#include_directories(${Pangolin_INCLUDE_DIRS})
#include_directories(${TOON_INCLUDE_PATHS})

add_subdirectory(${PROJECT_SOURCE_DIR}/src)