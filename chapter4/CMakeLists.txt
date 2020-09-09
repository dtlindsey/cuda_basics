cmake_minimum_required(VERSION 3.17)

add_executable(cu_ch4 ch4.cu)

link_directories(../lib)
add_executable(cu_julia ch4.2.cu)
target_link_libraries(cu_julia glut64)
