// #include <iostream>
#include <stdio.h>

#include "./common/book.h"

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}

int main( void ) {
    int count;
    cudaDeviceProp prop;

    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Device count: %d\n", count);

    if (count){
        for (int i =0; i< count; i++) {
            HANDLE_ERROR(
                    cudaGetDeviceProperties(&prop, i)
            );
            printf("---------General Device Info for device %d--------------\n", i);
            printf("Name:    %s\n", prop.name);
            printf("Compute capability: %d.%d\n", prop.major, prop.minor);
            printf("Clock rate: %d\n", prop.clockRate);
            printf("Device copy overlap: ");
            if (prop.deviceOverlap) {
                printf("Enabled\n");
            } else {
                printf("Disabled\n");
            }
            printf("Kernel execution timeout: ");
            if (prop.kernelExecTimeoutEnabled) {
                printf("Enabled\n");
            } else {
                printf("Disabled\n");
            }
            printf("--- Memory Information for device %d ---\n", i);
            printf("Total global mem: %1d\n", prop.totalGlobalMem);
            printf("Total constant mem: %1d\n", prop.totalConstMem);
            printf("Max mem pitch: %1d\n", prop.memPitch);
            printf("Texture Alignment: %1d\n", prop.textureAlignment);
        }
    }

    return 0;
}
