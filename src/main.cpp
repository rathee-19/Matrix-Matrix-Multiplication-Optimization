#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <immintrin.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <omp.h>

namespace solution
{
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
    {
        // Check for valid matrix dimensions for multiplication
        if (k != m)
        {
            std::cerr << "Error: Matrix dimensions are not compatible for multiplication." << std::endl;
            return "";
        }

        std::string sol_path = (std::filesystem::temp_directory_path() / "student_sol.dat").string();

        int sol_fd = open(sol_path.c_str(), O_RDWR | O_CREAT, (mode_t)0600);
        size_t sol_size = sizeof(float) * n * m;

        // Extend the file size to accommodate the result matrix
        if (ftruncate(sol_fd, sol_size) == -1)
        {
            std::cerr << "Error: Failed to extend file size: " << strerror(errno) << std::endl;
            close(sol_fd);
            return "";
        }

        // Map the output file into memory for writing
        float *result = static_cast<float *>(mmap(NULL, sol_size, PROT_WRITE, MAP_SHARED, sol_fd, 0));
        if (result == MAP_FAILED)
        {
            std::cerr << "Error: mmap failed: " << strerror(errno) << std::endl;
            close(sol_fd);
            return "";
        }

        int m1_fd = open(m1_path.c_str(), O_RDONLY);
        size_t m1_size = sizeof(float) * n * k;
        float *mat1 = static_cast<float *>(mmap(NULL, m1_size, PROT_READ, MAP_SHARED, m1_fd, 0));

        int m2_fd = open(m2_path.c_str(), O_RDONLY);
        size_t m2_size = sizeof(float) * k * m;
        float *mat2 = static_cast<float *>(mmap(NULL, m2_size, PROT_READ, MAP_SHARED, m2_fd, 0));

        const int block_size = 64;
        const int vector_size = 8;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int block_i = 0; block_i < m; block_i += block_size)
        {
            for (int block_j = 0; block_j < n; block_j += block_size)
            {
                for (int col_A = block_i; col_A < block_i + block_size && col_A < m; ++col_A)
                {
                    for (int row_down_A = block_j; row_down_A < block_j + block_size && row_down_A < n; ++row_down_A)
                    {
                        __m256 r = _mm256_set1_ps(mat1[col_A * n + row_down_A]);

                        for (int row_B = 0; row_B < m; row_B += vector_size)
                        {
                            // Prefetch data for the next iteration
                            _mm_prefetch(reinterpret_cast<const char*>(&mat2[(row_down_A + 1) * m + row_B]), _MM_HINT_T0);

                            // Load 8 elements from mat2 into AVX register
                            __m256 a = _mm256_loadu_ps(&mat2[row_down_A * m + row_B]);
                            // Load 8 elements from result into AVX register
                            __m256 b = _mm256_loadu_ps(&result[col_A * m + row_B]);
                            // Multiply and accumulate
                            __m256 c = _mm256_fmadd_ps(r, a, b);
                            // Store result back to memory
                            _mm256_storeu_ps(&result[col_A * m + row_B], c);
                        }
                    }
                }
            }
        }

        // Unmap memory-mapped files
        munmap(mat1, m1_size);
        munmap(mat2, m2_size);
        munmap(result, sol_size);

        // Close file descriptors
        close(m1_fd);
        close(m2_fd);
        close(sol_fd);

        return sol_path;
    }
}
