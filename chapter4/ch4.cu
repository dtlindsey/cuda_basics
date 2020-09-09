#include <stdio.h>
#include <time.h>

#include "../common/book.h"

#define NUM_THREADS 10

// cuda
__global__ void add(int *a, int *b, int *c) {
    int t_id = blockIdx.x;
    if (t_id < NUM_THREADS) {
        c[t_id] = a[t_id] * b[t_id];
        printf("GPU %d * %d = %d\n",a[t_id], b[t_id], c[t_id]);
    }
}

int main( void ) {
    int a[NUM_THREADS], b[NUM_THREADS], c[NUM_THREADS];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( &dev_a, NUM_THREADS * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( &dev_b, NUM_THREADS * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( &dev_c, NUM_THREADS * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<NUM_THREADS; i++) {
        a[i] = -i;
        b[i] = i * i;
        c[i] = -100;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, NUM_THREADS * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, NUM_THREADS * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add<<<NUM_THREADS,1>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, NUM_THREADS * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<NUM_THREADS; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}
//
//int main(void) {
//    int a[NUM_THREADS], b[NUM_THREADS], c[NUM_THREADS];
//    int *dev_a, *dev_b, *dev_c;
//
////    time_t start, end;
//    // add data on cpu
////    start = time(0x0);
//
//    HANDLE_ERROR(cudaMalloc((void** ) &dev_a,
//                            (NUM_THREADS * sizeof(int))));
//    HANDLE_ERROR(cudaMalloc((void** ) &dev_b,
//                            (NUM_THREADS * sizeof(int))));
//    HANDLE_ERROR(cudaMalloc((void** ) &dev_c,
//                            (NUM_THREADS * sizeof(int))));
//
//    // fill the array
//    for (int i = 0; i<NUM_THREADS; i++) {
//        a[i] = -1;
//        b[i] = i * i;
//    }
//
//    HANDLE_ERROR(cudaMemcpy(dev_a,
//                            a,
//                            (NUM_THREADS * sizeof(int)),
//                            cudaMemcpyHostToDevice));
//    HANDLE_ERROR(cudaMemcpy(dev_b,
//                            b,
//                            (NUM_THREADS * sizeof(int)),
//                            cudaMemcpyHostToDevice));
//
//    add<<<NUM_THREADS, 1>>>(dev_a, dev_b, dev_c);
////    printf("testing %d", dev_c[0]);
//    HANDLE_ERROR(cudaMemcpy(c,
//                            dev_c,
//                            (NUM_THREADS * sizeof(int)),
//                            cudaMemcpyDeviceToHost));
//
////    end = time(0x0) - start;
//
//    for (int i = 0; i < NUM_THREADS; i++) {
//        printf("%d * %d = %d\n",a[i], b[i], c[i]);
//    }
////    printf("Took %ld seconds to compute\n",end);
//
//    HANDLE_ERROR(cudaFree(dev_a));
//    HANDLE_ERROR(cudaFree(dev_b));
//    HANDLE_ERROR(cudaFree(dev_c));
//
//    return 0;
//}
