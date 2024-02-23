#cmake_minimum_required(VERSION 3.15)
cmake_minimum_required(VERSION 3.8)
project(GR_ADMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "mpic++")

add_executable(GR_ADMM allreduce/common.h allreduce/p2p_communicator.h allreduce/reduceoperator.h allreduce/ringallreduce.h allreduce/simpleallreduce.h src/admm_lr_function.cpp
        include/admm_lr_function.h utils/utils.cpp utils/utils.h src/neighbors.cpp include/neighbors.h src/sparse_dataset.cpp include/sparse_dataset.h src/string_util.cpp include/string_util.h src/conf_util.cpp include/conf_util.h include/type_convert.h include/simple_algebra.h src/properties.cpp include/properties.h src/prob.cpp include/prob.h include/optimizer.h src/group_admm.cpp include/group_admm.h src/lr_tron_optimizer.cpp include/lr_tron_optimizer.h
        src/train.cpp src/simple_algebra.cpp src/gradient_decent_optimizer.cpp include/gradient_decent_optimizer.h include/differentiable_function.h src/differentiable_function.cpp)