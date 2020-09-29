cmake_minimum_required(VERSION 3.17)
project(cuda_test LANGUAGES CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 -lcudart")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# first app
add_executable(cu_main main.cu)
# chapter 4
add_subdirectory(chapter4)
# chapter 5
add_subdirectory(chapter5)


