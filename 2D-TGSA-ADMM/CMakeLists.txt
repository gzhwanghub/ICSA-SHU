cmake_minimum_required(VERSION 3.8)
project(sparse_admm)


set(CMAKE_CXX_COMPILER "mpic++")
#SET(CMAKE_C_COMPILER  mpicc)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILE)
add_executable(sparse_admm ${SRC_FILE})
