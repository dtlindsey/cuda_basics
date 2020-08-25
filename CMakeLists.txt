cmake_minimum_required(VERSION 3.17)
project(cuda_test CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cu_device_info main.cu)

add_executable(cu_add ch4.cu)

link_directories(./lib)
add_executable(cu_julia ch4.2.cu)
target_link_libraries(cu_julia glut64)

set_target_properties(
        cu_device_info
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
