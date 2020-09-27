#include <stdio.h>

#include "cuda.h"
#include "../common/book.h"

#define DIM 1024

#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
        imin( 32,(N + threadsPerBlock -1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set cache value
    cache[cacheIndex] = temp;

    // sync
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cacheIndex[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
}

int main( void ) {
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc( N * sizeof(float));
    partial_c = (float*) malloc( blocksPerGrid * sizeof(float));

    // allocate mem
    HANDLE_ERROR(cudaMalloc((void**) &dev_a,
                            N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_b,
                            N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**) &dev_partial_c,
                            blocksPerGrid * sizeof(float)));

    // fill host
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>> (dev_a,
                                             dev_b,
                                             dev_partial_c);

}


