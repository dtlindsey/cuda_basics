// #include <iostream>
#include <stdio.h>

#include "./common/book.h"

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}

int main( void ) {
    cudaDeviceProp prop;
    int device;

    HANDLE_ERROR(cudaGetDevice(&device));
    printf("ID of current device: %d\n", device);

    memset(&prop, 0, sizeof(cudaDeviceProp));

    prop.major = 7;
    prop.minor = 1;

    HANDLE_ERROR(cudaChooseDevice(&device, &prop));

    printf("ID of device with revisions of data 7.1 or above: %d\n", device);

    HANDLE_ERROR(cudaSetDevice(device));

    return 0;
}
