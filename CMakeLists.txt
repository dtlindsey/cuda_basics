cmake_minimum_required(VERSION 3.17)
project(cuda_test CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_test main.cu)

set_target_properties(
        cuda_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
