# Tiled-MM

This is a library for multiplying matrices on GPU. As opposed to NVIDIA's `cublas`, this library takes pointer from the host side (CPU), performs tiling and then copies and multiplies tile by tile on a GPU.

## Performance

The benchmarks were performed on a single node of Piz Daint Supercomputer (Cray XC50), equipped with a `P100` NVIDIA GPU. We compared the performance of our library `Tiled-MM` with the vanilla version of `cublasXt` and also with the manually tuned version of `cublasXt`, where we manually set the tile size to `4000` and enabled the pinned memory mode. `Tiled-MM` was substantially faster than the vanilla version of `cublasXt`, and achieved similar performance as the manually tuned version of `cublasXt`, as can be seen from the results below.
<p align="center"><img src="https://github.com/kabicm/Tiled-MM/blob/master/docs/performance.svg" width="90%"></p>

In the benchmark, we used `double precision`, `square matrices` given in `column-major` ordering, and `alpha = beta = 1.0`.

## Features:

- The user can specify the tile size of each dimension separately.
- The user can specify the number of streams to be used.
- The user can reuse the same context (and thus the same device memory) for many multiplications which can lead to significant performance improvements.
- Fully templatized, supporting arbitrary data types.
- Ported to both `NVIDIA` and `AMD` GPUs.

## Limitations

These are just current limitations which are planned to be handled in some future release.
- At the moment supports only the column-major ordering of `A, B` and `C`, but this can be easily improved.

## Building and Installing

Assuming that you want to use the `gcc 8` compiler, you can build the project as follows:
```bash
# clone the repo
git clone --recursive https://github.com/kabicm/Tiled-MM
cd Tiled-MM
mkdir build
cd build

# build
CC=gcc-8 CXX=g++-8 cmake -DTILEDMM_GPU_BACKEND=CUDA ..

# compile
make -j 4
```

The option `-DTILEDMM_GPU_BACKEND` can have the following values:
- `CUDA`: for NVIDIA GPUs
- `ROCM`: for AMD GPUs

## Example

Using the library is very simple, just include `#include <tiled_mm.hpp>` and use it as follows:
```cpp
// A dimensions: MxK
auto a_host = gpu::malloc_device<double>(M*K, 1);
// B dimensions: KxN
auto b_host = gpu::malloc_device<double>(K*N, 1);
// C dimensions: MxN
auto c_host = gpu::malloc_device<double>(M*N, 0);

double alpha = 1.0;
double beta = 1.0;

// preallocates device buffers and other CUDA stuff
// the context does not have to be created explicitly
// so the user can omit this part
auto ctx = gpu::make_context();

// compute c = alpha * a * b + beta * c
// There is also a version without ctx, in case the user
// does not want to create the context explicitly
gpu::dgemm(ctx, a_host, b_host, c_host, M, N, K, alpha, beta); // optional
```
When creating the context, the user can specify tile dimensions and the number of streams to be used as:
```cpp
int tile_size_m = 4000;
int tile_size_n = 4000;
int tile_size_k = 4000;
int n_streams = 4;

auto ctx = gpu::make_context(n_streams, tile_size_m, tile_size_n, tile_size_k);
```

After compilation, there is a small example application that can be run from the build folder as follows:
```bash
./examples/multiply -m 10000 -n 10000 -k 10000 -r 1
```
Where flags have the following meaning:
- `m`: Number of rows of `A` and `C`.
- `n`: Number of columns of `A` and `C`.
- `k`: Number of columns of `A` and rows of `B`.
- `r`: Number of repetitions.

## Testing

The result is compared against `cublasXt`. 
The tests can be run inside the build folder with:
```bash
make test
```

## Author
Marko Kabic (marko.kabic@cscs.ch)
