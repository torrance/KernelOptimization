#pragma once

#include <array>
#include <tuple>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#define HIPCHECK(res) { hipcheck(res, __FILE__, __LINE__); }

inline void hipcheck(hipError_t res, const char* file, int line) {
    if (res != hipSuccess) throw std::runtime_error("Fatal hipError");
}

template <typename T>
struct GridSpec {
    long N;
    T deltalm;

    // Convert a linear index into a NxN grid into 2D coordinates, scaled by deltalm
    __host__ __device__ inline auto linearToSky(const size_t idx) const {
        long lpx = idx / N;
        long mpx = idx % N;
        return std::make_tuple((lpx - N / 2) * deltalm, (mpx - N / 2) * deltalm);
    }

    __host__ __device__ size_t size() const { return N * N; }
};

template <typename T>
using Matrix2x2 = std::array<T, 4>;

template <typename T>
using ComplexMatrix2x2 = std::array<thrust::complex<T>, 4>;

template <typename T>
__host__ __device__ inline
std::array<T, 4> matmul(const std::array<T, 4>& lhs, const std::array<T, 4>& rhs) {
    return {
        lhs[0] * rhs[0] + lhs[1] * rhs[2],
        lhs[0] * rhs[1] + lhs[1] * rhs[3],
        lhs[2] * rhs[0] + lhs[3] * rhs[2],
        lhs[2] * rhs[1] + lhs[3] * rhs[3]
    };
}

template <typename T>
__host__ __device__ inline
ComplexMatrix2x2<T> conjugate_transpose(const ComplexMatrix2x2<T>& A) {
    return {thrust::conj(A[0]), thrust::conj(A[2]), thrust::conj(A[1]), thrust::conj(A[3])};
}

template <typename T>
__host__ __device__ inline T ndash(const T l, const T m) {
    T r2 = std::min<T>(l*l + m*m, T(1));
    return r2 / (1 + sqrt(1 - r2));
}

__device__ inline auto cispi(const float& theta) {
    float real, imag;
    sincospif(theta, &imag, &real);
    return thrust::complex(real, imag);
}

__device__ inline auto cispi(const double& theta) {
    double real, imag;
    sincospi(theta, &imag, &real);
    return thrust::complex(real, imag);
}

template <typename T>
__host__ __device__ auto cld(T x, T y) {
    return (x + y - 1) / y;
}