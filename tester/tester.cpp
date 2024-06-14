#include <iostream>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <exception>
#include <memory>
#include <fstream>
#include <immintrin.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <studentlib.h>

void __terminate_gracefully(const std::string &msg) noexcept {
	std::cout << -1 << std::endl;
	std::cerr << msg << std::endl;
	exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]){
	#if defined(__cpp_lib_filesystem)
		// Header available
	#else
		__terminate_gracefully("<filesystem> header is not supported by this compiler");
	#endif
	try{
		// Parse arguments
		if(argc < 3) __terminate_gracefully("Usage: ./tester.out <n> <k> <m> <optional:seed>");	
		std::random_device rd;
		std::mt19937 rng(argc > 4 ? std::atoi(argv[4]) : rd());
		std::int32_t n = std::atoi(argv[1]), k = std::atoi(argv[2]), m = std::atoi(argv[3]);
		// Util func
		std::function<float(void)> generateRandomfloat = [&](){
			static std::uniform_real_distribution<float> distribution(0.0, 100.0);
			return distribution(rng);
		};
		// Create psuedo-bitmap file
		std::string m1_path = std::filesystem::temp_directory_path() / ("1-" + std::to_string(n) + "x" + std::to_string(k) + ".dat");
		std::string m2_path = std::filesystem::temp_directory_path() / ("2-" + std::to_string(k) + "x" + std::to_string(m) + ".dat");
		std::cout << "[1/5] Looking for input file(s)" << std::endl;
		auto gen_mat = [&](std::string &file_path, int num_rows, int num_cols){
			auto mat = std::make_unique<float[]>(num_rows * num_cols);
			if(std::filesystem::exists(file_path)) {
				std::cout << "\t- Input file: " << file_path << " found, using existing input file" << std::endl;
				std::ifstream in_fs(file_path, std::ios::binary);
				in_fs.read(reinterpret_cast<char*>(mat.get()), sizeof(float) * num_rows * num_cols);
			}
			else{
				std::cout << "\t- Input file not found. Creating new test data: " << file_path << std::endl;
				std::ofstream out_fs(file_path, std::ios::binary);
				for(std::int32_t i=0; i < num_rows * num_cols; i++) mat[i] = generateRandomfloat();
				out_fs.write(reinterpret_cast<char*>(mat.get()), sizeof(float) * num_rows * num_cols);
			}
			return std::move(mat);
		};
		auto m1 = gen_mat(m1_path, n, k), m2 = gen_mat(m2_path, k, m);
		// Create solution_file
		std::string sol_path = std::filesystem::temp_directory_path() / ("sol-" + std::to_string(n) + "x" + std::to_string(m) + ".dat");
		std::cout << "[2/5] Looking for verification file " << sol_path << std::endl;
		if(std::filesystem::exists(sol_path)) std::cout << "[3/5] Verification file found, using existing verification data" << std::endl;
		else{
			std::cout << "[3/5] Verification file not found. Creating new verification data" << std::endl;
			auto result = std::make_unique<float[]>(n*m);
		    for (int i = 0; i < n; i++)
		        for (int j = 0; j < m; j++) {
		            result[i*m + j] = 0;
		            for (int l = 0; l < k; ++l) 
		            	result[i*m + j] += m1[i*k + l] * m2[l*m + j];
		        }
			std::ofstream sol_fs(sol_path, std::ios::binary);
			sol_fs.write(reinterpret_cast<char*>(result.get()), sizeof(float) * n * m);
		}
		m1.reset(); m2.reset();
		std::cout << "[4/5] Running student solution" << std::endl;
		// Time your solution's execution time
		auto start = std::chrono::high_resolution_clock::now();
		const std::string student_sol_path = solution::compute(m1_path, m2_path, n, k, m);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		std::cout << "[5/5] Verifying student solution" << std::endl;
		// Verify solution
		std::int32_t fd_sol = open(sol_path.c_str(), O_RDONLY);
		std::int32_t fd_student = open(student_sol_path.c_str(), O_RDONLY);
		const std::size_t file_size = n * m;
		const auto sol_data = reinterpret_cast<float*>(mmap(nullptr, file_size * sizeof(float), PROT_READ, MAP_PRIVATE, fd_sol, 0));
		const auto student_data = reinterpret_cast<float*>(mmap(nullptr, file_size * sizeof(float), PROT_READ, MAP_PRIVATE, fd_student, 0));
		constexpr const float error_threshold = 1e-3;
		std::uint32_t remaining = 0;
		#ifdef __AVX512F__
			const __m512 threshold_vec = _mm512_set1_ps(error_threshold);
			for (std::uint32_t i = 0; i + 16 <= file_size; i += 16) {
				__m512 abs_diff_vec = _mm512_abs_ps(_mm512_sub_ps(_mm512_loadu_ps(sol_data + i), _mm512_loadu_ps(student_data + i)));
				__mmask16 mask = _mm512_cmp_ps_mask(abs_diff_vec, threshold_vec, _CMP_GT_OQ);
				if(mask){
					float error = _mm512_mask_reduce_max_ps(mask, abs_diff_vec);
					__terminate_gracefully("Solution has higher error than required: " + std::to_string(error));
				}
			}
			remaining = file_size % 16;
		#else
			const __m256 threshold_vecp = _mm256_set1_ps(error_threshold);
			const __m256 threshold_vecn = _mm256_set1_ps(-error_threshold);
			for (std::uint32_t i = 0; i+8 <= file_size; i += 8) {
				__m256 diff_vec = _mm256_sub_ps(_mm256_loadu_ps(sol_data + i), _mm256_loadu_ps(student_data + i));
				__m256 maskp = _mm256_cmp_ps(diff_vec, threshold_vecp, _CMP_GT_OQ);
				__m256 maskn = _mm256_cmp_ps(diff_vec, threshold_vecn, _CMP_LT_OQ);
				std::int32_t mask = _mm256_movemask_ps(_mm256_or_ps(maskp, maskn));
				if(mask) __terminate_gracefully("Solution has higher error than required.");
			}
			remaining = file_size % 8;
		#endif
		for(std::uint32_t i=file_size-remaining; i < file_size; i++){
			float diff = std::abs(sol_data[i] - student_data[i]);
			if(diff > error_threshold) __terminate_gracefully("Solution has higher error than required: " + std::to_string(diff));
		}
		munmap(sol_data, file_size * sizeof(float)); munmap(student_data, file_size * sizeof(float));
		close(fd_sol); close(fd_student);
		std::filesystem::remove(student_sol_path);
		std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
	
	} catch(const std::exception &e){
		__terminate_gracefully(e.what());
	}
	return EXIT_SUCCESS;
}
