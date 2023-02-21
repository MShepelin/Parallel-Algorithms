#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDA-By-Example/common/book.h"
#include "CUDA-By-Example/common/cpu_bitmap.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdint.h>

#define GRID_DIM 1000

#define RANK_SEARCH_FLAGS_SIZE 1

#define PAIRS_PER_ROUND 65536
#define BLOCKS_FOR_PAIRS_SEARCH 256
#define INVALID_PAIR_VALUE -1

#define cudaCheckError(msg) {  \
	cudaError_t __err = cudaGetLastError();  \
	if(__err != cudaSuccess) {  \
		fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
					(msg), cudaGetErrorString(__err), \
					__FILE__, __LINE__); \
		fprintf(stderr, "*** FAILED - ABORTING\n"); \
		exit(1); \
	} \
}

// TODO: change type in subtraction_pairs for uint32_t
__global__ void perform_subtractions(
	const int32_t* subtraction_pairs,
	const int32_t* input_columns_offsets,
	const int32_t* input_column_sizes,
	const int32_t* input_rows_indicies,
	const int32_t* output_columns_offsets,
	int32_t* output_column_sizes,
	int32_t* output_rows_indicies) {
	// Assumes subtraction_pairs has size (PAIRS_PER_ROUND * 2)
	for (int32_t pair_id = blockIdx.x * blockDim.x + threadIdx.x; pair_id < PAIRS_PER_ROUND; pair_id += gridDim.x * blockDim.x) {
		int32_t column_from = subtraction_pairs[pair_id * 2];
		int32_t column_subtraction = subtraction_pairs[pair_id * 2 + 1];

		uint32_t id_to_put = output_columns_offsets[column_from];
		uint32_t left_column_id = input_columns_offsets[column_from];
		const uint32_t left_column_id_limit = left_column_id + input_column_sizes[column_from];
		uint32_t right_column_id = input_columns_offsets[column_subtraction];
		const uint32_t right_column_id_limit = right_column_id + input_column_sizes[column_subtraction];

		while (left_column_id < left_column_id_limit ||
			right_column_id < right_column_id_limit) {
			
			uint32_t left_low = (left_column_id < left_column_id_limit) ? input_rows_indicies[left_column_id] : UINT32_MAX;
			uint32_t right_low = (right_column_id < right_column_id_limit) ? input_rows_indicies[right_column_id] : UINT32_MAX;

			if (left_low == right_low) {
				++left_column_id;
				++right_column_id;
				// 1 ^ 1 = 0
			}
			else if (left_low < right_low) {
				output_rows_indicies[id_to_put] = left_low;
				++output_column_sizes[column_from];
				++id_to_put;
				++left_column_id;
				// 1 ^ 0 = 1
			}
			else if (right_low < left_low) {
				output_rows_indicies[id_to_put] = right_low;
				++output_column_sizes[column_from];
				++id_to_put;
				++right_column_id;
				// 0 ^ 1 = 1
			}
		}
	}
}

__global__ void find_subtraction_pairs_raw(int32_t* nnz_estimation, int32_t* subtraction_pairs, int32_t* column_sizes, uint32_t columns) {
	// Assumes subtraction_pairs has size (PAIRS_PER_ROUND * 2)
	
	// Each block has N threads
	// Each thread works for a unique column and 
	// checks all columns with lower indexes (starting from left)

	// Assumes memory_calculation has size gridDim.x

	__shared__ int32_t new_subtraction_id;

	if (threadIdx.x == 0) {
		new_subtraction_id = 0;
	}

	__syncthreads();
	
	uint32_t max_subtractions = PAIRS_PER_ROUND / gridDim.x;
	uint32_t offset = blockIdx.x * max_subtractions;
	// TODO: add assertion that PAIRS_PER_ROUND % gridDim.x == 0
	// TODO: may be make them static

	for (size_t column_id = blockIdx.x * blockDim.x + threadIdx.x; column_id < columns; column_id += gridDim.x * blockDim.x) {
		bool is_subtraction_found = false;

		if (column_sizes[column_id] > 0) {
			for (size_t left_column_id = 0; left_column_id < column_id; ++left_column_id) {
				if (column_sizes[column_id] == column_sizes[left_column_id]) {
					int32_t old_new_subtraction_id = atomicAdd(&new_subtraction_id, 1);
					if (old_new_subtraction_id >= max_subtractions) {
						// Block batch is full
						break;
					}

					is_subtraction_found = true;
					nnz_estimation[column_id] = column_sizes[column_id] + column_sizes[left_column_id] - 2;
					subtraction_pairs[(offset + old_new_subtraction_id) * 2] = column_id;
					subtraction_pairs[(offset + old_new_subtraction_id) * 2 + 1] = left_column_id;
					// subtraction pair means columns[column_id] -= columns[left_column_id]
					break;
				}
			}
		}

		if (!is_subtraction_found) {
			// No atomic operations are needed because 
			// each column_id is devoted to one thread
			nnz_estimation[column_id] = column_sizes[column_id];
		}
	}
}

__global__ void check_if_matrix_reduced_raw(
	int32_t* rank_search_flags,
	int32_t* column_sizes, 
	uint32_t columns) {
	// Check every pair of (i, j) where i and j are column indicies
	size_t columns_pairs = columns * columns;
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < columns_pairs; i += gridDim.x * blockDim.x) {
		size_t column_left = i % columns;
		size_t column_right = i - column_left * columns;
		if (column_sizes[column_left] == column_sizes[column_right]) {
			atomicOr(rank_search_flags, 1);
		}
	}
}

__global__ void fill_column_sizes(int32_t* column_sizes, uint32_t columns, int32_t* columns_offsets) {
	// Assumes columns_offsets has size of (columns + 1)
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < columns; i += gridDim.x * blockDim.x) {
		column_sizes[i] = columns_offsets[i + 1] - columns_offsets[i];
	}
}

struct CSRMatrix {
public:
	thrust::device_vector<int32_t> d_columns_offsets;
	thrust::device_vector<int32_t> d_rows_indicies;
	// Number of real elements in column,
	// is <= (difference in d_columns_offsets neighbour elements)
	thrust::device_vector<int32_t> d_column_sizes; 

public:
	CSRMatrix() = delete;

	CSRMatrix(int32_t columns) {
		d_column_sizes.assign(columns, 0);
		d_columns_offsets.assign(columns + 1, -1);
		// We put invalid size value
		// TODO: check that d_columns_offsets really has size (columns + 1)
	}

	CSRMatrix(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns) {
		d_columns_offsets.assign(column_offsets, column_offsets + column_offsets_len);
		d_rows_indicies.assign(rows_indicies, rows_indicies + nnz);
		d_column_sizes.assign(columns, 0);
		fill_column_sizes<<<256, 256>>>(
			thrust::raw_pointer_cast(d_column_sizes.data()), d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data())); // TODO: fix grid size
	}

	void check_if_matrix_reduced(thrust::device_vector<int32_t>& rank_search_flags) {
		check_if_matrix_reduced_raw<<<256, 256>>>( // TODO: fix grid size
			thrust::raw_pointer_cast(rank_search_flags.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size());
	}

	void find_subtraction_pairs(
		thrust::device_vector<int32_t>& d_nnz_estimation,
		thrust::device_vector<int32_t>& d_pairs_for_subtractions) {
		find_subtraction_pairs_raw<<<BLOCKS_FOR_PAIRS_SEARCH, 256>>>(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size()
		);
	}

	// TODO: add squash method to remove all garbage data in d_rows_indicies

	void perform_subtraction(CSRMatrix& output, const thrust::device_vector<int32_t>& d_pairs_for_subtractions) const {
		perform_subtractions<<<256, 256>>>(
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),

			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data())
		);
	}

	void update_columns_offsets(thrust::device_vector<int32_t>& d_nnz_estimation) {
		// TODO: figure out a better way to update columns offsets
		thrust::host_vector<int32_t> nnz_estimation = d_nnz_estimation;
		thrust::host_vector<int32_t> new_columns_offsets;
		new_columns_offsets.assign(d_column_sizes.size() + 1, 0);

		for (size_t i = 1; i < d_column_sizes.size() + 1; ++i) {
			new_columns_offsets[i] = new_columns_offsets[i - 1] + nnz_estimation[i - 1];
		}

		d_columns_offsets = new_columns_offsets;
		d_rows_indicies.assign(new_columns_offsets[d_column_sizes.size()], -1);
		d_column_sizes.assign(d_column_sizes.size(), 0);
	}
};

extern "C" void read_CSR(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns, int32_t rows) {
	CSRMatrix buffers[] = {
		CSRMatrix(column_offsets, column_offsets_len, rows_indicies, nnz, columns),
		CSRMatrix(columns)
	};
	uint32_t active_buffer_index = 0;
	
	thrust::device_vector<int32_t> rank_search_flags(RANK_SEARCH_FLAGS_SIZE, false);
	// Structure of rank_search_flags:
	// 0) is matrix reduced?

	thrust::device_vector<int32_t> d_pairs_for_subtractions(PAIRS_PER_ROUND * 2, -1);
	thrust::device_vector<int32_t> d_nnz_estimation(columns, 0);

	cudaCheckError("Buffer initialisation");

	// Do while not reduced:
	for (int32_t attempt = 0; (attempt < 1) && (!rank_search_flags[0]); ++attempt) {
		// TODO: figure out a better way to check boolean
		// TODO: define maimum attempts or take it from function arguements
		d_pairs_for_subtractions.assign(PAIRS_PER_ROUND * 2, INVALID_PAIR_VALUE);
		buffers[active_buffer_index].find_subtraction_pairs(d_nnz_estimation, d_pairs_for_subtractions);
		// TODO: check that values (from) don't repeat in pairs value
		// TODO: check that all columns are set in d_nnz_estimation
		buffers[1 - active_buffer_index].update_columns_offsets(d_nnz_estimation);
		// perform subtraction with merge
		buffers[active_buffer_index].perform_subtraction(buffers[1 - active_buffer_index], d_pairs_for_subtractions);

		active_buffer_index = 1 - active_buffer_index;
		buffers[active_buffer_index].check_if_matrix_reduced(rank_search_flags);
		cudaCheckError("Matrix reduction check");

		// [IMPORTANT] check that algorithm works when -1 can be found in rows_indicies (extra memory space) and column_size (empty columns)
	}
}
