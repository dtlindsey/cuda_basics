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
            printf(" --- MP Information for device %d ---\n", i);
            printf("Multiprocessor count: %d\n",
                   prop.multiProcessorCount);
            printf("Shared mem per mp: %1d\n", prop.sharedMemPerBlock);
            printf("Registers per mp: %d\n", prop.regsPerBlock);
            printf("Threads in warp: %d\n", prop.warpSize);
            printf("Max threads per block: %d\n",
                   prop.maxThreadsPerBlock);
            printf("Max thread dimensions: (%d, %d, %d)\n",
                   prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                   prop.maxThreadsDim[2]);
            printf("Max grid dimensions: (%d, %d, %d)\n",
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("\n");
        }
    }

    return 0;
}
