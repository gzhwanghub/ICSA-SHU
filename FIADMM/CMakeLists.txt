cmake_minimum_required(VERSION 3.8)
project(FIADMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(${PROJECT_SOURCE_DIR} /include /allreduce)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILE)
add_executable(FIADMM ${SRC_FILE})

