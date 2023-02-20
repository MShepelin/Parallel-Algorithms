#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDA-By-Example/common/book.h"
#include "CUDA-By-Example/common/cpu_bitmap.h"
#include <thrust/device_vector.h>

#define GRID_DIM 1000

#define RANK_SEARCH_FLAGS_SIZE 1

#define PAIRS_PER_ROUND 65536
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

__global__ void find_subtraction_pairs_raw(int32_t* subtraction_pairs, int32_t* d_column_sizes, uint32_t columns) {
	// Assumes subtraction_pairs has size (PAIRS_PER_ROUND * 2)
	
	// Each block has N threads
	// Each thread works for a unique column and 
	// checks all columns with lower indexes (starting from left)

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
		for (size_t left_column_id = 0; left_column_id < columns; ++left_column_id) {
			if (d_column_sizes[column_id] == d_column_sizes[left_column_id]) {
				int32_t old_new_subtraction_id = atomicAdd(&new_subtraction_id, 1);
				if (old_new_subtraction_id >= max_subtractions) {
					// Block batch is full
					break;
				}

				subtraction_pairs[(offset + old_new_subtraction_id) * 2] = column_id;
				subtraction_pairs[(offset + old_new_subtraction_id) * 2 + 1] = left_column_id;
				// subtraction pair means columns[column_id] -= columns[left_column_id]
				break;
			}
		}
	}
}

__global__ void check_if_matrix_reduced_raw(
	int32_t* rank_search_flags,
	int32_t* d_column_sizes, 
	uint32_t columns) {
	// Check every pair of (i, j) where i and j are column indicies
	size_t columns_pairs = columns * columns;
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < columns_pairs; i += gridDim.x * blockDim.x) {
		size_t column_left = i % columns;
		size_t column_right = i - column_left * columns;
		if (d_column_sizes[column_left] == d_column_sizes[column_right]) {
			atomicOr(rank_search_flags, 1);
		}
	}
}

__global__ void fill_column_sizes(int32_t* d_column_sizes, uint32_t columns, int32_t* d_columns_offsets) {
	// Assumes d_columns_offsets has size of (columns + 1)
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < columns; i += gridDim.x * blockDim.x) {
		d_column_sizes[i] = d_columns_offsets[i + 1] - d_columns_offsets[i];
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
	CSRMatrix() = default;

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

	void find_subtraction_pairs(thrust::device_vector<int32_t>& d_pairs_for_subtractions) {
		find_subtraction_pairs_raw<<<256, 256>>>(
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size()
		);
	}
};

extern "C" void read_CSR(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns, int32_t rows) {
	CSRMatrix buffers[] = {
		CSRMatrix(column_offsets, column_offsets_len, rows_indicies, nnz, columns),
		CSRMatrix()
	};
	uint32_t active_buffer_index = 0;

	
	thrust::device_vector<int32_t> rank_search_flags(RANK_SEARCH_FLAGS_SIZE, false);
	// Structure of rank_search_flags:
	// 0) is matrix reduced?

	thrust::device_vector<int32_t> d_pairs_for_subtractions(PAIRS_PER_ROUND * 2, -1);
	cudaCheckError("Buffer initialisation");

	// Do while not reduced:
	while (!rank_search_flags[0]) { // TODO: figure out a better way to check boolean
		d_pairs_for_subtractions.assign(PAIRS_PER_ROUND * 2, INVALID_PAIR_VALUE);
		buffers[active_buffer_index].check_if_matrix_reduced(rank_search_flags);
		buffers[active_buffer_index].find_subtraction_pairs(d_pairs_for_subtractions);

		active_buffer_index = 1 - active_buffer_index;
		cudaCheckError("Matrix reduction check");
		break;
	}
}
