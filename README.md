# Matrix-Matrix Multiplication Optimization Project

## Project Overview

This project focuses on optimizing the performance of matrix-matrix multiplication operations on large datasets. The primary goal is to leverage various optimization techniques to significantly reduce computation time and enhance efficiency. This is crucial for applications in scientific computing, graphics, machine learning, and other domains requiring high-performance numerical computations.

## Main File Analysis and Optimizations

The primary optimizations in the `main.cpp` file include:

### Compiler Optimizations

- **Optimization Level `O3`**: This is the highest optimization level for the GCC compiler, enabling aggressive optimizations like inlining, loop unrolling, and vectorization. These optimizations aim to reduce the execution time by improving the efficiency of the generated machine code.

### Targeted CPU Features

- **AVX2**: Advanced Vector Extensions 2 provides 256-bit SIMD (Single Instruction, Multiple Data) operations, allowing parallel processing of data. This significantly speeds up arithmetic operations on large datasets.
- **BMI, BMI2, LZCNT, POPCNT**: These are Bit Manipulation Instructions, Leading Zero Count, and Population Count instructions for efficient bitwise operations. They enhance the performance of certain algorithmic operations that rely heavily on bit manipulation.

### Memory Mapping

- **Memory-Mapped Files**: The use of memory-mapped files allows for efficient reading and writing of large matrices without loading the entire dataset into RAM at once. This is particularly useful for handling very large files.
- **`mmap`**: This system call maps files or devices into memory, facilitating large-scale data manipulation directly in the address space of the process.

### Parallelization

- **OpenMP**: OpenMP is utilized to parallelize the matrix multiplication process across multiple CPU cores. This significantly reduces computation time by dividing the workload among several threads.

### SIMD Intrinsics

- **AVX Intrinsics**: These intrinsics are used to perform matrix multiplication operations in parallel, leveraging the SIMD capabilities of modern CPUs for higher performance. The use of intrinsics allows for fine-grained control over CPU instructions, optimizing performance for specific hardware.

### Blocked Matrix Multiplication

- **Blocking/Tiling**: This technique improves cache performance by dividing matrices into smaller sub-blocks (tiles) that fit into the CPU cache. The block size used in this project is 64x64. Blocking helps to reduce cache misses and enhances data locality, leading to improved performance.

### Specific Optimization Details

### 1. Blocking

Blocking or tiling is a technique used to increase the cache hit rate by dividing the matrices into smaller sub-blocks. In this implementation, a block size of 64x64 is used. This size is chosen to fit well within the L1 cache of modern CPUs, ensuring that each sub-block can be processed efficiently without frequent cache misses.

```cpp
cppCopy code
const int block_size = 64;

```

### 2. SIMD Vectorization

SIMD vectorization is achieved using AVX2 intrinsics, which allow processing of 8 floating-point numbers simultaneously. This provides a significant performance boost for the matrix multiplication.

```cpp
cppCopy code
const int vector_size = 8;
#pragma omp parallel for collapse(2) schedule(static)
for (int block_i = 0; block_i < m; block_i += block_size) {
    for (int block_j = 0; block_j < n; block_j += block_size) {
        for (int col_A = block_i; col_A < block_i + block_size && col_A < m; ++col_A) {
            for (int row_down_A = block_j; row_down_A < block_j + block_size && row_down_A < n; ++row_down_A) {
                __m256 r = _mm256_set1_ps(mat1[col_A * n + row_down_A]);

                for (int row_B = 0; row_B < m; row_B += vector_size) {
                    _mm_prefetch(reinterpret_cast<const char*>(&mat2[(row_down_A + 1) * m + row_B]), _MM_HINT_T0);
                    __m256 a = _mm256_loadu_ps(&mat2[row_down_A * m + row_B]);
                    __m256 b = _mm256_loadu_ps(&result[col_A * m + row_B]);
                    __m256 c = _mm256_fmadd_ps(r, a, b);
                    _mm256_storeu_ps(&result[col_A * m + row_B], c);
                }
            }
        }
    }
}

```

- **Prefetching**: `_mm_prefetch` is used to load data into the cache before it is needed, reducing cache miss penalties.
- **Loading Data**: `_mm256_loadu_ps` loads 8 floating-point values into an AVX register.
- **Fused Multiply-Add**: `_mm256_fmadd_ps` performs a fused multiply-add operation, combining multiplication and addition into a single instruction, which is more efficient than performing them separately.
- **Storing Data**: `_mm256_storeu_ps` stores 8 floating-point values from an AVX register back to memory.

## Conclusion

This project demonstrates significant performance improvements for matrix-matrix multiplication operations on large datasets through the use of advanced optimization techniques.