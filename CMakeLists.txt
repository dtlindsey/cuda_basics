cmake_minimum_required(VERSION 3.17)
project(cuda_test LANGUAGES CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(
        CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_72,code=sm_72 \
        -gencode arch=compute_75,code=sm_75 \
        -lcudart"
        )
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# first app
add_executable(cu_main main.cu)
# chapter 4
add_subdirectory(chapter4)
# chapter 5
add_subdirectory(chapter5)


