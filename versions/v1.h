#pragma once

#include <array>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include "../common.h"

template <typename T>
__global__ void v1(
    thrust::complex<T>* grid,
    const T* us, const T* vs, const T* ws,
    const ComplexMatrix2x2<T>* data, const Matrix2x2<T>* weights, const ComplexMatrix2x2<T>* As,
    const GridSpec<T> gridspec, const size_t N
) {
    for (
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < gridspec.size();
        i += blockDim.x * gridDim.x
    ) {
        // Find the sky coordinates (l,m,n) corresponding to grid[idx]
        auto [l, m] = gridspec.linearToSky(i);
        T n { ndash(l, m) };

        // Accumulation value
        ComplexMatrix2x2<T> cell {};

        // Iterate over the data and perform the reduction
        for (size_t j {}; j < N; ++j) {
            // exp {2i * pi * (ul + vm + wn)}
            thrust::complex<T> phase { cispi(2 * (us[j] * l + vs[j] * m + ws[j] * n)) };

            // Retrieve weight and datum matrices
            auto weight = weights[j];
            auto datum = data[j];

            // Multiply phase * datum * weight piecewise
            cell[0] += phase * datum[0] * weight[0];
            cell[1] += phase * datum[1] * weight[1];
            cell[2] += phase * datum[2] * weight[2];
            cell[3] += phase * datum[3] * weight[3];
        }

        // Finally, write back to global memory by applying A-terms
        // sum(A * cell * A_H), where A_H is the conjugate transpose
        auto A = As[i];
        auto A_H = conjugate_transpose(A);
        cell = matmul(matmul(A, cell), A_H);
        grid[i] = cell[0] + cell[1] + cell[2] + cell[3];
    }
}