# Matrix-Matrix Multiplication Optimization Project

## Project Overview

This repo focuses on optimizing the performance of matrix-matrix multiplication operations on large datasets. The primary goal is to leverage various optimization techniques to significantly reduce computation time and enhance efficiency. This is crucial for applications in scientific computing, graphics, machine learning, and other domains requiring high-performance numerical computations.

## Main File Analysis and Optimizations

The primary optimizations in the `main.cpp` file include:

### Compiler Optimizations

- **Optimization Level `O3`**: The highest optimization level for the GCC compiler, enabling aggressive optimizations like inlining, loop unrolling, and vectorization.
- **Unroll Loops (`funroll-loops`)**: Unroll loops to reduce the overhead of loop control and increase instruction-level parallelism.

### Targeted CPU Features

- **AVX2**: Advanced Vector Extensions 2, which provides 256-bit SIMD (Single Instruction, Multiple Data) operations, allowing for parallel processing of data.
- **BMI, BMI2, LZCNT, POPCNT**: Bit Manipulation Instructions, Leading Zero Count, and Population Count instructions for efficient bitwise operations.

### Memory Mapping

- **Memory-Mapped Files**: Used for reading large matrices from files into memory, allowing efficient access and modification.
- **`mmap`**: Maps files or devices into memory, facilitating large-scale data manipulation without loading the entire file into RAM.

### Parallelization

- **OpenMP**: Utilized to parallelize the matrix multiplication process across multiple CPU cores, significantly reducing computation time.

### SIMD Intrinsics

- **AVX Intrinsics**: Used to perform matrix multiplication operations in parallel, leveraging the SIMD capabilities of modern CPUs for higher performance.

## How to Build and Run

### Prerequisites

- GCC Compiler
- CMake
- OpenMP

### Build Instructions

1. **Clone the repository**:
    
    ```bash
    bashCopy code
    git clone <repository-url>
    cd <repository-directory>
    
    ```
    

### Run the Application

To run the matrix-matrix multiplication application, use the provided `runner_script.sh`:

## Conclusion

This project demonstrates significant performance improvements for matrix-matrix multiplication operations on large datasets through the use of advanced optimization techniques. The combination of compiler optimizations, manual vectorization, and parallelization yields substantial gains, making it feasible to handle large-scale data efficiently.