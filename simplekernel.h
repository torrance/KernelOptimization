#include <hip/hip_runtime.h>

__global__ void _simplekernel(float* data, size_t N) {
    for (
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x
    ) {
        float datum {};
        for (size_t j {}; j < 1000; ++j) {
            for (size_t k {}; k < 1000; ++k) {
                datum += j * k;
            }
        }
        data[i] = datum;
    }
}

void simplekernel() {
    const size_t N {1024 * 128};

    float* data;
    HIPCHECK( hipMalloc(&data, sizeof(float) * N) );

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    double duration {};

    for (size_t i {}; i < 1; ++i) {
        HIPCHECK( hipEventRecord(start, hipStreamPerThread) );
        hipLaunchKernelGGL(
            _simplekernel, 1024, 128, 0, hipStreamPerThread, data, N
        );
        HIPCHECK( hipEventRecord(stop, hipStreamPerThread) );
        HIPCHECK( hipEventSynchronize(stop) );

        float thisduration;
        HIPCHECK( hipEventElapsedTime(&thisduration, start, stop) );
        duration += thisduration;
    }
    std::cout << "Simple Kernel elapsed: " << duration / 100 << std::endl;
}