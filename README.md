# A Tale of 2 GPUs

## Introduction

This is really a plea for help. Why, oh why, is my kernel so poorly performant on AMD GPUs, specifically the MI250X, when it performs so very well on a mid-level NVIDIA A100.

Some background: I'm a radio astronomer, and last year our local supercomputer recently switched to using AMD MI250X accelerators. I was given a grant to rewrite some algorithms to use these new accelerators, which in practice meant diving into the world of AMD's 'hip' ecosystem. If you don't know what that is, it's essentially a thin abstraction over both CUDA and ROCm backends, using (almost) the same functions you already know from CUDA, and allowing you to target either platform using a compile time switch.

As a case study, I want to present what is a rather simple kernel, and my attempts to increase performance. But also my failure to get the MI250X to reach its stated performance. Secretly, I'm hoping someone will come along and show me the stupid thing I'm doing that is holding back the performance of my code.

## The problem

This particular kernel is used in imaging radio astronomy data. Radio interferometers use interference patterns ("correlations") between pairs of antenna ("baselines") within the array to measure the radio sky. The relation between this so-called 'visibility' domain and the image domain is essentially just a Fourier transform. The problem is, these correlations are sparse, spatially irregular (i.e. not gridded), and subject to all sorts of time varying effects, and the size of the radio astronomy data can be very large. And there's a whole field of algorithms to make this imaging problem compuationally tractable.

**The details don't matter for today.** What matters is that one particular algorithm to solve this problem ([image domain gridding](https://arxiv.org/abs/1909.07226); IDG) leans heavily on the following little sum. 

$$
\begin{aligned}
f\left(l, m\right) &= \sum_{i} I_i \exp{ 2\pi i \left( u_il + v_im + w_i (1 - \sqrt{1 - l^2 - m^2} \right) } \\
&= \sum_{i} I_i \exp{ 2\pi i \left( u_il + v_im + w_i n' \right) }
\end{aligned}
$$

In this case, $(l, m)$ are the sky coordinates. You can think of them as the pixel values. And the index $i$ represents a sum over the full set of data, where $(u, v, w)$ are spatial coordinates of a baseline, and $I$  (a $2 \times 2$ complex valued matrix) is the interference pattern of that baseline at some instant in time. So in words: for every pixel in our $N \times N$ grid, we want to compute this sum over the full set of data.

To add one small complication to the picture, we also apply a so-called A-term correction, which is a $2 \times 2$ complex valued matrix that helps correct for beam effects, ionospheric effects, what-have-you, for that specific position on the sky. This correction sandwiches the sum, multiplying on the left using standard matrix multiplication, and likewise on the right with its conjugate transpose:

$$
\begin{aligned}
f\left(l, m\right) = A_{lm} \left[ \sum_{i} I_i \exp{ 2\pi i \left( u_il + v_im + w_i n' \right) } \right] A_{lm}^\dagger
\end{aligned}
$$

That's mostly it. This computation is made many times as part of the IDG algorithm on a bunch of low-resolution grids, usually on something like $128 \times 128$ pxiels, and then using some mathematical sleight of hand, all these low-resolution subgrids can be combined to construct a final high resolution image. 

This summation is a bottleneck, and it's important it runs as fast as possible. So let's get started.

## Version 0

To ease us into this, let's start things on the CPU. The function signaure, which we'll use for all our kernels, looks like this:

* The inout value `grid` is a our complex valued $N \times N$ map of the sky.
* The summation over $j$ (up to `N`) occurs over each of `u`, `v`, `w`, which are simple scalar coordinates, either float or double, and over each of the $2 \times 2$ `weights` and `data` matrices (implemented here as `std::array<T, 4>`).
* Meanwhile, the `gridspec` object just encodes the dimension of the grid, and helps us transfrom from pixel coordinates to $(l, m)$.

The body of the function first iterates over each of the `grid` pixels, indexed by `i`, and then in turn computes the summation over the the full set of data, index by `j`:

Some small notes:

* Recall Euler's formula, whereby $\exp(i \theta) = \cos(\theta) + i \sin(\theta)$. We use this relation to compute the exponential here, since the exponent is purely imaginary, that's what those trig functions are doing.
* We're using `thrust:::complex` as the complex implementation here. It's virtually identical to to `std::complex` but is well supported on the GPU, and even seems to be more performant in some situations.
* The `weights` matrix is a simple, real valued $2 \times 2$ matrix that we apply _piecewise_ to the data.
* The little `#pragma omp parallel for` directive parallelizes each pixel sum over all available CPU cores using OpenMP.

I think that's all there is to comment on this. Hopefully it reads relatively straightforwardly.

As a baseline benchmark, on a 32 (64 HT) core machine, this takes about 9.7 seconds.

## Aside: benchmarking

Throughout we're going to be benchmarking these kernels. The benchmarking code is available [here](XXX), and all times are given as the mean over 100 samples for the following configuration:

* A $128 \times 128$ grid
* Using 1 million data points, i.e. $N = 1,000,000$

For the GPU code, I reset the memory before each kernel run to ensure the L1 cache isn't giving us an artifical boost for subsequent samples.

The GPU devices I am benchmarking against are the following:

| **GPU**               | **Software**       | Peak FP32 [TFLOPS] |
| --------------------- | ------------------ | ------------------ |
| NVIDIA A100-SXM4-40GB | CUDA Version 12.4  | 19.49              |
| AMD Radeon PRO W6800  | ROCm Version 6.1.0 | 17.82              |
| AMD MI250X            | ROCm Version 6.1.0 | 47.87 (23.94*)     |

(The little asterisk next the the AMD MI250X is because each of these cards presents two logical GPU units, and so _I assume_ the performance we should expect from a single unit is half the stated value.)

The peak FP32 is a stated performance for these devices, and should definitely be taken with a grain of salt, but it should give us some sense of the relative performance we should expect. These are all very similarly specced, but all things being equal, we should expect the AMD MI250X to come out on top. (Foreshadowing).

## Version 1

Let's pump out a straightforward GPU implementation of the kernel. Just as we did on the CPU, we're going to parallelize over each of the subgrids, and let each GPU thread sum over the full dataset and write out to its respective grid pixel.

The only thing worth commenting on here is that I've used a `cispi()` function for computing the `phase` value. This is a small utility function I've written that internally uses [`sincospif`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1gaab8978300988c385e0aa4b6cba44225e) or [`sincospi`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#group__cuda__math__double_1gafc99d7acfc1b14dcb6f6db56147d2560), depending on flaoting point precision, to compute both the real and imaginary components of the exponential at once, for a small performance boost.

Even though this is a very basic implementation of the kernel, we already see a large performance increase over the CPU.