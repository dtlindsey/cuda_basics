cmake_minimum_required(VERSION 3.17)

add_executable(threaded_add ch5.2.cu)

link_directories(../lib)
add_executable(gpu_ripple gpu_ripple.cu)
target_link_libraries(cu_julia glut64)
