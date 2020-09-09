#include <stdio.h>

#include "../common/book.h"

#define N (30 * 1024)

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate memory on GPU
    HandleError(cudaMalloc(&dev_a, N * sizeof(int)), "cudaMalloc dev_a", 20);
    HandleError(cudaMalloc(&dev_b, N * sizeof(int)), "cudaMalloc dev_b", 21);
    HandleError(cudaMalloc(&dev_c, N * sizeof(int)), "cudaMalloc dev_c", 22);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy data
    HandleError(cudaMemcpy(dev_a,
                           a,
                           N * sizeof(int),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy dev_a", 29);
    HandleError(cudaMemcpy(dev_b,
                           b,
                           N * sizeof(int),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy dev_b", 34);

    add<<<128,128>>>(dev_a, dev_b, dev_c);

    // copy result back to host
    HandleError(cudaMemcpy(c,
                           dev_c,
                           N * sizeof(int),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy to host dev_c", 43);

    // display results
    for (int i=0; i < N; ++i) {
        if (a[i] + b[i] == c[i]) {
            printf("%d + %d = %d\n", a[i], b[i], c[i]);
        }
        else {
            printf("%d + %d != %d\n", a[i],b[i],c[i]);
        }
    }

    HandleError(cudaFree(dev_a), "cudaFree dev_a", 54);
    HandleError(cudaFree(dev_b), "cudaFree dev_b", 55);
    HandleError(cudaFree(dev_c), "cudaFree dev_c", 56);
    //
    return 0;
}
