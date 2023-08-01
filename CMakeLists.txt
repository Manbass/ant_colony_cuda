cmake_minimum_required(VERSION 3.25)
project(cuda_ant_colony CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

add_executable(cuda_ant_colony main.cu)

set_target_properties(cuda_ant_colony PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
