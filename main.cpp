#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "common.h"
#include "versions/v0.h"
#include "versions/v1.h"
#include "versions/v2.h"
#include "versions/v3.h"
#include "versions/v4.h"
#include "versions/v5.h"
#include "versions/v6.h"

#define HIPCHECK(res) { hipcheck(res, __FILE__, __LINE__); }

inline void hipcheck(hipError_t res, const char* file, int line) {
    if (res != hipSuccess) throw std::runtime_error("Fatal hipError");
}

void fft() {
    hipfftComplex* data;
    HIPCHECK( hipMalloc(&data, sizeof(hipfftComplex) * 16000 * 16000) );

    hipfftComplex* rubbish;
    HIPCHECK( hipMallocHost((void**) &rubbish, sizeof(hipfftComplex) * 16000 * 16000) );

    hipfftHandle plan;
    hipfftPlan2d(&plan, 16000, 16000, HIPFFT_C2C);
    hipfftSetStream(plan, hipStreamPerThread);

    hipEvent_t start, stop;
    HIPCHECK( hipEventCreate(&start) );
    HIPCHECK( hipEventCreate(&stop) );

    double duration;
    for (size_t i {}; i < 100; ++i) {
        HIPCHECK( hipMemcpy(data, rubbish, sizeof(hipfftComplex) * 16000 * 16000, hipMemcpyHostToDevice) );

        hipEventRecord(start, hipStreamPerThread);
        hipfftExecC2C(plan, data, data, HIPFFT_FORWARD);
        hipEventRecord(stop, hipStreamPerThread);

        hipEventSynchronize(stop);

        float thisduration;
        hipEventElapsedTime(&thisduration, start, stop);
        duration += thisduration;
    }

    std::cout << "FFT elapsed: " << duration / 100 << std::endl;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipfftDestroy(plan);
    HIPCHECK( hipFree(data) );
}

template <typename T>
void benchmark(auto fn, dim3 blocks, dim3 threads) {
    const size_t N {100000}; // {1000000};
    const size_t samples {1};
    double duration {};

    GridSpec<T> gridspec {128, 0.01};

    // Allocate data on host
    std::vector<thrust::complex<T>> grid_h(gridspec.size());
    std::vector<T> us_h(N), vs_h(N), ws_h(N);
    std::vector<Matrix2x2<T>> weights_h(N);
    std::vector<ComplexMatrix2x2<T>> data_h(N), As_h(gridspec.size());

    // Populate with random data
    std::mt19937 gen(1234);
    std::uniform_real_distribution<T> randf(-1, 1);

    for (size_t n {}; n < N; ++n) {
        us_h[n] = randf(gen);
        vs_h[n] = randf(gen);
        ws_h[n] = randf(gen);
        weights_h[n] = {randf(gen), randf(gen), randf(gen), randf(gen)};
        data_h[n] = {
            thrust::complex(randf(gen), randf(gen)), thrust::complex(randf(gen), randf(gen)),
            thrust::complex(randf(gen), randf(gen)), thrust::complex(randf(gen), randf(gen))
        };
    }
    for (size_t i {}; i < gridspec.size(); ++i) {
        As_h[i] = {
            thrust::complex(randf(gen), randf(gen)), thrust::complex(randf(gen), randf(gen)),
            thrust::complex(randf(gen), randf(gen)), thrust::complex(randf(gen), randf(gen))
        };
    }

    // Run CPU version to validate GPU correctness
    std::cout << "Starting CPU..." << std::endl;
    double cpuduration {};
    {
        auto start = std::chrono::steady_clock::now();
        v0(
            grid_h.data(), us_h.data(), vs_h.data(), ws_h.data(),
            data_h.data(), weights_h.data(), As_h.data(), gridspec, N
        );
        auto stop = std::chrono::steady_clock::now();
        cpuduration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        cpuduration /= 1000; // convert to ms
    }
    std::cout << "CPU duration: " << cpuduration << " ms" << std::endl;

    // Allocate memory on the device
    T *us, *vs, *ws;
    HIPCHECK( hipMalloc(&us, sizeof(T) * N) );
    HIPCHECK( hipMalloc(&vs, sizeof(T) * N) );
    HIPCHECK( hipMalloc(&ws, sizeof(T) * N) );

    thrust::complex<T>* grid;
    HIPCHECK( hipMalloc(&grid, sizeof(thrust::complex<T>) * gridspec.size()) );

    Matrix2x2<T>* weights;
    HIPCHECK( hipMalloc(&weights, sizeof(Matrix2x2<T>) * N) );

    ComplexMatrix2x2<T> *data, *As;
    HIPCHECK( hipMalloc(&data, sizeof(ComplexMatrix2x2<T>) * N) );
    HIPCHECK( hipMalloc(&As, sizeof(ComplexMatrix2x2<T>) * gridspec.size()) );

    hipEvent_t start, stop;
    HIPCHECK( hipEventCreate(&start) );
    HIPCHECK( hipEventCreate(&stop) );

    for (size_t i {}; i < samples; ++i) {
        HIPCHECK( hipMemcpy(us, us_h.data(), sizeof(T) * N, hipMemcpyHostToDevice) );
        HIPCHECK( hipMemcpy(vs, vs_h.data(), sizeof(T) * N, hipMemcpyHostToDevice) );
        HIPCHECK( hipMemcpy(ws, ws_h.data(), sizeof(T) * N, hipMemcpyHostToDevice) );
        HIPCHECK( hipMemcpy(weights, weights_h.data(), sizeof(Matrix2x2<T>) * N, hipMemcpyHostToDevice) );
        HIPCHECK( hipMemcpy(data, data_h.data(), sizeof(ComplexMatrix2x2<T>) * N, hipMemcpyHostToDevice) );
        HIPCHECK( hipMemcpy(As, As_h.data(), sizeof(ComplexMatrix2x2<T>) * gridspec.size(), hipMemcpyHostToDevice) );

        HIPCHECK( hipMemset(grid, 0, sizeof(thrust::complex<T>) * gridspec.size()) );

        HIPCHECK( hipEventRecord(start, hipStreamPerThread) );
        hipLaunchKernelGGL(
            fn, blocks, threads, 0, hipStreamPerThread,
            grid, us, vs, ws, data, weights, As, gridspec, N
        );
        HIPCHECK( hipEventRecord(stop, hipStreamPerThread) );
        HIPCHECK( hipEventSynchronize(stop) );

        float thisduration;
        HIPCHECK( hipEventElapsedTime(&thisduration, start, stop) );
        duration += thisduration;

        // Validate output
        std::vector<thrust::complex<T>> result(gridspec.size());
        HIPCHECK( hipMemcpy(result.data(), grid, sizeof(thrust::complex<T>) * gridspec.size(), hipMemcpyDeviceToHost) );
        for (size_t j {}; j < gridspec.size(); ++j) {
            if (thrust::abs(result[j] - grid_h[j]) / (thrust::abs(result[j]) + thrust::abs(grid_h[j])) > 1e-3) {
                std::cout << "Error: (pixel=" << j << ") " << thrust::abs(result[j] - grid_h[j]) << std::endl;
                throw std::runtime_error("GPU and CPU results did not agree");
            }
        }
    }

    HIPCHECK( hipFree(us) );
    HIPCHECK( hipFree(vs) );
    HIPCHECK( hipFree(ws) );
    HIPCHECK( hipFree(grid) );
    HIPCHECK( hipFree(weights) );
    HIPCHECK( hipFree(data) );
    HIPCHECK( hipFree(As) );
    HIPCHECK( hipEventDestroy(start) );
    HIPCHECK( hipEventDestroy(stop) );

    duration /= samples;
    std::cout << "Elapsed: " << duration << " ms" << std::endl;
}

int main() {
    fft();
    benchmark<float>(v1<float>, dim3(128), dim3(128));
    benchmark<float>(v2<float>, dim3(128), dim3(128));
    benchmark<float>(v3<float>, dim3(128), dim3(128));
    benchmark<float>(v4<float>, dim3(128, 8), dim3(128, 1));
    benchmark<float>(v5<float>, dim3(128, 8), dim3(128, 1));
}