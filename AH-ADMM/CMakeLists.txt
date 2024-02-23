cmake_minimum_required(VERSION 3.9.5)
project(admm_end)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_FLAGS -fopenmp)
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories (${PROJECT_SOURCE_DIR}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILE)


add_executable(admm_end ${SRC_FILE})
