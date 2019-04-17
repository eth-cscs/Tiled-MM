# Tiled-MM

This is a library for multiplying matrices on GPU. As opposed to NVIDIA's `cublas`, this library takes pointer from the host side (CPU), performs tiling and then copies and multiplies tile by tile on a GPU.

## Performance

When tested on `P100` GPU against NVIDIA's `cublasXt`, this library outperforms by more than 60\% in some cases, as can be seen in the following plot:
<p align="center"><img src="https://github.com/kabicm/Tiled-MM/blob/master/docs/performance.svg" width="80%"></p>
The main improvement comes from the fact that Tiled-MM preallocates the device buffers in the moment of the context-creation and then keeps reusing it, whereas cublasXt seems to allocate and deallocate device buffers on the fly.

In the benchmark, we used double precision, `column-major` ordering of matrices, and `alpha = beta = 1.0`.

## Features:

- The user can specify the tile size of each dimension separately.
- The user can specify the number of CUDA streams to be used.
- The user can reuse the same context (and thus the same device memory) for many multiplications which can lead to significant performance improvements.

## Limitations

These are just current limitations which are planned to be handled in some future release.
- At the moment supports only the column-major ordering of `A, B` and `C`, but this can be easily improved.
- At the moment, only double precision is supported

## Building and Installing

Assuming that you want to use the `gcc 8` compiler, you can build the project as follows:
```bash
# clone the repo
git clone https://github.com/kabicm/Tiled-MM
cd Tiled-MM
mkdir build
cd build

# build
CC=gcc-8 CXX=g++-8 cmake -DCMAKE_BUILD_TYPE=Release ..

# compile
make -j 4
```

## Example

Using the library is very simple, just include `#include <tiled_mm.hpp>` and use it as follows:
```cpp
// A dimensions: MxK
auto a_host = gpu::malloc_pinned<value_type>(M*K, 1);
// B dimensions: KxN
auto b_host = gpu::malloc_pinned<value_type>(K*N, 1);
// C dimensions: MxN
auto c_host = gpu::malloc_pinned<value_type>(M*N, 0);

double alpha = 1.0;
value_type beta = 1.0;

// preallocates device buffers and other CUDA stuff
auto ctx = gpu::make_context();

// compute c = alpha * a * b + beta * c
gpu::dgemm(ctx, a_host, b_host, c_host, M, N, K, alpha, beta);
```
When creating context, a user can specify tile dimensions and the number of streams to be used as:
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
