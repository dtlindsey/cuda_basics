#include <stdio.h>
#include <time.h>

#include "./common/book.h"
#include "./common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2 (void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator* (const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+ (const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};


__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (((float)DIM/2 - x) / (DIM/2));
    float jy = scale * (((float)DIM/2 - y) / (DIM/2));
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i=0; i < 350; i++) {
        a = a * a + c;
        if (a.magnitude2() > DIM) {
            return 0;
        }
    }
    return 1;
}

__global__ void kernel (unsigned char *ptr) {
    // map block.idx
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    //
    int juliaValue = julia(y, x);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 255 * juliaValue;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
    unsigned  char *dev_bitmap;
};

int main(void) {
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
//    unsigned char *ptr = bitmap.get_ptr();
    unsigned char *dev_bitmap;

    cudaError_t did_init = cudaMalloc( (void**) &dev_bitmap,bitmap.image_size());
    if(did_init != cudaSuccess) {
        printf("Data didn't get initialized\n");
        return 1;
    }

    data.dev_bitmap = dev_bitmap;

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_bitmap);

    cudaError_t did_copy = cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                                bitmap.image_size(),
                                cudaMemcpyDeviceToHost);

    if (did_copy != cudaSuccess) {
        printf("Didn't copy to host\n");
        return 1;
    }

    bitmap.display_and_exit();

    bool did_free = cudaFree(dev_bitmap);
    if(!did_free) {
        printf("didn't free gpu memory\n");
        return 1;
    }

    return 0;
}

