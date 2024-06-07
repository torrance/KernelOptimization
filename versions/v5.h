#pragma once

#include <array>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include "../common.h"

template <typename T>
__global__ void v5(
    thrust::complex<T>* grid,
    const T* us, const T* vs, const T* ws,
    const ComplexMatrix2x2<T>* data, const Matrix2x2<T>* weights, const ComplexMatrix2x2<T>* As,
    const GridSpec<T> gridspec, const size_t N
) {
    // Set up shared memory to cache uvw, data and cache values
    const size_t cachesize {128};
    __shared__ char cache[
        cachesize * sizeof(std::array<T, 3>) +
        cachesize * sizeof(ComplexMatrix2x2<T>)
    ];

    auto uvwcache = reinterpret_cast<std::array<T, 3>*>(cache);
    auto datacache = reinterpret_cast<ComplexMatrix2x2<T>*>(uvwcache + cachesize);

    for (
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < blockDim.x * cld<size_t>(gridspec.size(), blockDim.x);
        i += blockDim.x * gridDim.x
    ) {
        // Find the sky coordinates (l,m,n) corresponding to grid[idx]
        auto [l, m] = gridspec.linearToSky(i);
        T n { ndash(l, m) };

        // Accumulation value
        ComplexMatrix2x2<T> cell {};

        // Loop over data and perform reduction in chunks of $cachesize
        for (size_t j {blockIdx.y * cachesize}; j < N; j += (gridDim.y * cachesize)) {
            // Handle remainder where N is not divisible by $cachesize
            const size_t cachebound = min(cachesize, N - j);

            // Populate the cache
            for (size_t k = threadIdx.x; k < cachebound; k += blockDim.x) {
                uvwcache[k] = {us[j + k], vs[j + k], ws[j + k]};
                datacache[k] = data[j + k];

                // Preapply weights
                auto weight = weights[j + k];
                auto& datum = datacache[k];
                datum[0] *= weight[0];
                datum[1] *= weight[1];
                datum[2] *= weight[2];
                datum[3] *= weight[3];
            }
            __syncthreads();

            // Iterate over the data and perform the reduction over the cache
            for (size_t k {}; k < cachebound; ++k) {
                // exp {2i * pi * (ul + vm + wn)}
                auto [u, v, w] = uvwcache[k];
                thrust::complex<T> phase { cispi(2 * (u * l + v * m + w * n)) };

                // Retrieve datum matrix
                auto datum = datacache[k];

                // Multiply phase * datum piecewise
                cell[0].real(cell[0].real() + phase.real() * datum[0].real() - phase.imag() * datum[0].imag());
                cell[0].imag(cell[0].imag() + phase.real() * datum[0].imag() + phase.imag() * datum[0].real());
                cell[1].real(cell[1].real() + phase.real() * datum[1].real() - phase.imag() * datum[1].imag());
                cell[1].imag(cell[1].imag() + phase.real() * datum[1].imag() + phase.imag() * datum[1].real());
                cell[2].real(cell[2].real() + phase.real() * datum[2].real() - phase.imag() * datum[2].imag());
                cell[2].imag(cell[2].imag() + phase.real() * datum[2].imag() + phase.imag() * datum[2].real());
                cell[3].real(cell[3].real() + phase.real() * datum[3].real() - phase.imag() * datum[3].imag());
                cell[3].imag(cell[3].imag() + phase.real() * datum[3].imag() + phase.imag() * datum[3].real());
            }
            __syncthreads();
        }

        // Finally, write back to global memory by applying A-terms
        // sum(A * cell * A_H), where A_H is the conjugate transpose
        if (i < gridspec.size()) {
            auto A = As[i];
            auto A_H = conjugate_transpose(A);
            cell = matmul(matmul(A, cell), A_H);

            auto result = cell[0] + cell[1] + cell[2] + cell[3];
            atomicAdd(reinterpret_cast<T*>(grid + i), result.real());
            atomicAdd(reinterpret_cast<T*>(grid + i) + 1, result.imag());
        }
    }
}