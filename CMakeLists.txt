cmake_minimum_required(VERSION 3.17)
project(cuda_test LANGUAGES CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

add_executable(cu_main main.cu)

add_executable(cu_ch4 ch4.cu)

link_directories(./lib)
add_executable(cu_julia ch4.2.cu)
target_link_libraries(cu_julia glut64)
