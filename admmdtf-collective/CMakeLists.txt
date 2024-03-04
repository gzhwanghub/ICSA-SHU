cmake_minimum_required(VERSION 3.8)
project(train_framework)

set(CMAKE_CXX_STANDARD 11)

#寻找PYTHONHOME的环境变量
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)



find_package (Python3 COMPONENTS Interpreter Development)
set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_C_COMPILER  mpicc)
include_directories(${Python3_INCLUDE_DIRS})
include_directories("/home/cluster/anaconda3/lib/python3.7/site-packages/mpi4py/include")
include_directories(${MPI_INCLUDE_PATH})


link_libraries(${Python3_LIBRARIES})

#指定你的.cxx等文件的目录
include_directories(${PROJECT_SOURCE_DIR}/src)

#寻找安装好的swig,其实就是去电脑中找你安装好的Swig环境，所以我们需要提前安装环境。

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
#Python文件输出目录 -outdir
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

#指定一个输出目录名，用于放置生成的源文件。例如输出的examplePTYHON_wrap.cxx文件等的位置
set(SWIG_OUTFILE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpp)#use "SWIG" with “CMake”  wrapping CPP to python

#注意这一步必须要加上，要不然不能编译c++，只能编译c
set_property(SOURCE src/train_framework.i PROPERTY CPLUSPLUS ON)


#开始生成
swig_add_library(train_framework
                LANGUAGE python
                SOURCES src/train_framework.i src/train_framework.cpp)



