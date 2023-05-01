#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <stdint.h>
#include <unordered_set>

#define RANK_SEARCH_FLAGS_SIZE 1

#define BLOCKS 65535
#define PAIRS_PER_ROUND BLOCKS * 8
#define THREADS 1024
#define REDUCE_STOP 8 // should be power of 2
#define SQUASHING_DELAY 1024

#define INVALID_PAIR_VALUE -1
#define COLUMN_STAYS_FIXED -1
#define INVALID_VALUE -1

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

template<typename T>
struct is_positive : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T& x) const
	{
		return x > T(0) ? 1 : 0;
	}
};

__device__ __inline__ int32_t get_mask_from_bool(int32_t a) {
	return ~(a - 1);
}

// TODO: change type in subtraction_pairs for uint32_t
__global__ void perform_subtractions(
	const int32_t* subtraction_pairs,
	const int32_t* input_columns_offsets,
	const int32_t* input_column_sizes,
	const int32_t* input_rows_indicies,
	const int32_t* output_columns_offsets,
	int32_t* output_column_sizes,
	int32_t* output_rows_indicies,
	int32_t* output_max_row_indexes) {
	// Assumes subtraction_pairs has size (PAIRS_PER_ROUND * 2)
	for (int32_t pair_id = blockIdx.x * blockDim.x + threadIdx.x; pair_id < PAIRS_PER_ROUND; pair_id += gridDim.x * blockDim.x) {
		const int32_t column_from = subtraction_pairs[pair_id * 2];
		const int32_t column_subtraction = subtraction_pairs[pair_id * 2 + 1];

		if (column_from == INVALID_PAIR_VALUE || column_subtraction == INVALID_PAIR_VALUE) {
			continue;
		}

		uint32_t id_to_put = output_columns_offsets[column_from];
		uint32_t left_column_id = input_columns_offsets[column_from];
		const uint32_t left_column_id_limit = left_column_id + input_column_sizes[column_from];
		uint32_t right_column_id = input_columns_offsets[column_subtraction];
		const uint32_t right_column_id_limit = right_column_id + input_column_sizes[column_subtraction];

		output_max_row_indexes[column_from] = 0;
		output_column_sizes[column_from] = 0;
		int32_t column_from_size = 0;
		while (left_column_id < left_column_id_limit || right_column_id < right_column_id_limit) {
			int32_t left_low = (left_column_id < left_column_id_limit) ? input_rows_indicies[left_column_id] : INT32_MAX;
			int32_t right_low = (right_column_id < right_column_id_limit) ? input_rows_indicies[right_column_id] : INT32_MAX;

#ifdef DEBUG_PRINT
			if (left_low == INVALID_VALUE || right_low == INVALID_VALUE) {
				printf("Error occured\n");
			}
#endif

			if (left_low == right_low) {
				++left_column_id;
				++right_column_id;
				// 1 ^ 1 = 0
			}
			else if (left_low < right_low) {
				output_rows_indicies[id_to_put] = left_low;
				++column_from_size;
				++id_to_put;
				++left_column_id;
				// 1 ^ 0 = 1
			}
			else if (right_low < left_low) {
				output_rows_indicies[id_to_put] = right_low;
				++column_from_size;
				++id_to_put;
				++right_column_id;
				// 0 ^ 1 = 1
			}
		}
		if (column_from_size > 0) {
			output_column_sizes[column_from] = column_from_size;
			output_max_row_indexes[column_from] = output_rows_indicies[id_to_put - 1];
		}
	}
}

__global__ void move_fixed_columns_raw(
	const int32_t* nnz_estimation,
	const int32_t* input_column_sizes,
	const int32_t* input_rows_indicies,
	const int32_t* input_columns_offsets,
	const int32_t* input_max_row_indexes,
	int32_t* output_column_sizes,
	int32_t* output_rows_indicies,
	const int32_t* output_columns_offsets,
	int32_t* output_max_row_indexes,
	uint32_t columns) {
	for (int32_t column_id = blockIdx.x * blockDim.x + threadIdx.x; column_id < columns; column_id += gridDim.x * blockDim.x) {
		if (nnz_estimation != nullptr && nnz_estimation[column_id] != COLUMN_STAYS_FIXED) {
			continue;
		}

		const uint32_t output_row_id = output_columns_offsets[column_id];
		const uint32_t input_row_id = input_columns_offsets[column_id];

		output_column_sizes[column_id] = input_column_sizes[column_id];
		output_max_row_indexes[column_id] = input_max_row_indexes[column_id];

#ifdef DEBUG_PRINT
		if (output_columns_offsets[column_id + 1] - output_row_id < input_column_sizes[column_id]) {
			printf("ALERT!!!");
		}
#endif

		for (uint32_t id_delta = 0; id_delta < input_column_sizes[column_id]; ++id_delta) {
			output_rows_indicies[output_row_id + id_delta] = input_rows_indicies[input_row_id + id_delta];
		}
	}
}

__global__ void find_subtraction_pairs_raw(int32_t* nnz_estimation, int32_t* subtraction_pairs, const int32_t* column_sizes, int32_t columns, const int32_t* columns_offset, const int32_t* rows_indices, const int32_t* max_row_indexes) {
	// Assumes subtraction_pairs has size (PAIRS_PER_ROUND * 2)
	
	// Each block has N threads
	// Each thread works for a unique column and 
	// checks all columns with lower indexes (starting from left)

	// Assumes memory_calculation has size gridDim.x
	__shared__ int32_t pivots[THREADS];
	__shared__ int32_t nnz[THREADS];
	
	// These variables are used only by one thread per block
	int32_t new_subtraction_id = 0;
	uint32_t max_subtractions = PAIRS_PER_ROUND / gridDim.x;
	uint32_t offset = blockIdx.x * max_subtractions;

	for (size_t column_id = blockIdx.x; column_id < columns; column_id += gridDim.x) {
		const int32_t column_size = column_sizes[column_id];

		if (column_size == 0) {
			if (threadIdx.x == 0) {
				nnz_estimation[column_id] = COLUMN_STAYS_FIXED;
			}
		}
		else {
			const int32_t max_row_index = max_row_indexes[column_id];
			pivots[threadIdx.x] = INVALID_VALUE;
			nnz[threadIdx.x] = INT32_MAX;

			// Find column with minimum number of nnz for each thread
			for (size_t column_compare_id = threadIdx.x; column_compare_id < columns; column_compare_id += blockDim.x) {
				const int32_t column_compare_size = column_sizes[column_compare_id];
				const int32_t comp_value = (max_row_indexes[column_compare_id] == max_row_index && column_compare_size < nnz[threadIdx.x]);
				const int32_t comp_value_mask = get_mask_from_bool(comp_value);

				nnz[threadIdx.x] = ((~comp_value_mask) & nnz[threadIdx.x]) | (comp_value_mask & column_compare_size);
				pivots[threadIdx.x] = ((~comp_value_mask) & pivots[threadIdx.x]) | (comp_value_mask & column_compare_id);
			}
			__syncthreads();

			uint32_t threads = THREADS;
			while (threads > REDUCE_STOP) {
				threads >>= 1;
				if (threadIdx.x < threads) {
					const int32_t column_compare_size = nnz[threadIdx.x + threads];
					const int32_t comp_value = (column_compare_size < nnz[threadIdx.x]);
					const int32_t comp_value_mask = get_mask_from_bool(comp_value);

					nnz[threadIdx.x] = ((~comp_value_mask) & nnz[threadIdx.x]) | (comp_value_mask & column_compare_size);
					pivots[threadIdx.x] = ((~comp_value_mask) & pivots[threadIdx.x]) | (comp_value_mask & pivots[threadIdx.x + threads]);
				}
				__syncthreads();
			}

			if (threadIdx.x == 0) {
				int32_t best_pivot = pivots[0];
				int32_t best_nnz = nnz[0];

				// Find column with minimum number of nnz among threads results
				for (size_t i = 1; i < REDUCE_STOP; ++i) {
					const int32_t comp_value = nnz[i] < best_nnz;
					const int32_t comp_value_mask = get_mask_from_bool(comp_value);
					best_nnz = (best_nnz & (~comp_value_mask)) | (comp_value_mask & nnz[i]);
					best_pivot = (best_pivot & (~comp_value_mask)) | (comp_value_mask & pivots[i]);
				}

				if (best_pivot != INVALID_VALUE && best_pivot != column_id && new_subtraction_id < max_subtractions) {
					// new_subtraction_id is unique for each block and is accessed only by first thread

					nnz_estimation[column_id] = column_size + best_nnz - 2;
					subtraction_pairs[(offset + new_subtraction_id) << 1] = column_id;
					subtraction_pairs[((offset + new_subtraction_id) << 1) + 1] = best_pivot;
					// subtraction pair means columns[column_id] -= columns[best_pivot]

					++new_subtraction_id;
				}
				else {
					// No atomic operations are needed because 
					// each column_id is devoted to one block
					nnz_estimation[column_id] = COLUMN_STAYS_FIXED; //column_sizes[column_id];
				}
			}
			__syncthreads();
		}
	}
}

__global__ void check_if_matrix_reduced_raw(
	int32_t* rank_search_flags,
	int32_t* column_sizes,
	uint32_t columns,
	const int32_t* column_offsets,
	int32_t* max_row_indexes,
	const int32_t* rows_indices) {
	
	for (size_t column_left = blockIdx.x; column_left < columns; column_left += gridDim.x) {
		if (column_sizes[column_left] <= 0) {
			continue;
		}

		const int32_t left_max_row_index = max_row_indexes[column_left];

		for (size_t column_right = column_left + threadIdx.x; column_right < columns; column_right += blockDim.x) {
			if (left_max_row_index == max_row_indexes[column_right]) {
				// column_sizes[column_right] is automatically > 0
				// Matrix is not reduced
				atomicAnd(&rank_search_flags[0], 0);
			}
		}
	}
}

__global__ void fill_column_sizes(int32_t* column_sizes, uint32_t columns, const int32_t* columns_offsets, int32_t* max_row_indexes, const int32_t* rows_indicies) {
	// Assumes columns_offsets has size of (columns + 1)
	// Assumes rows_indicies doesn't have fake values
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < columns; i += gridDim.x * blockDim.x) {
		const size_t next_column_offset = columns_offsets[i + 1];
		const size_t column_size = next_column_offset - columns_offsets[i];
		column_sizes[i] = column_size;
		
		if (column_size > 0 && next_column_offset >= 1) {
			max_row_indexes[i] = rows_indicies[next_column_offset - 1];
		}
		else {
			max_row_indexes[i] = INVALID_VALUE;
		}

#ifdef DEBUG_PRINT
		if (column_sizes[i] < 0) {
			printf("Initialised column_sizes with error\n");
			printf("i=%d\n", i);
			printf("columns_offsets[i + 1]=%d, columns_offsets[i]=%d", columns_offsets[i + 1], columns_offsets[i]);
		}
#endif
	}
}

struct CSRMatrix {
private:
	int32_t rows;
	thrust::device_vector<int32_t> d_columns_offsets;
	thrust::device_vector<int32_t> d_rows_indicies;
	// Number of real elements in column,
	// is <= (difference in d_columns_offsets neighbour elements)
	thrust::device_vector<int32_t> d_column_sizes; 
	thrust::device_vector<int32_t> d_max_row_indexes;

public:
	CSRMatrix() = delete;

	CSRMatrix(const int32_t in_columns, const int32_t in_rows) {
		d_column_sizes.assign(in_columns, 0);
		d_max_row_indexes.assign(in_columns, 0);
		// We put invalid size values in d_columns_offsets
		d_columns_offsets.assign(in_columns + 1, INVALID_VALUE);
		// d_rows_indicies stays empty
		
		// TODO: check that d_columns_offsets really has size (columns + 1)
		rows = in_rows;
	}

	CSRMatrix(const int32_t* column_offsets, const uint32_t column_offsets_len, const int32_t* rows_indicies, const uint32_t nnz, const int32_t in_columns, const int32_t in_rows, cudaStream_t stream) {
		d_column_sizes.assign(in_columns, 0);
		d_max_row_indexes.assign(in_columns, 0);
		d_columns_offsets.assign(column_offsets, column_offsets + column_offsets_len);
		d_rows_indicies.assign(rows_indicies, rows_indicies + nnz);
		fill_column_sizes<<<BLOCKS, THREADS, 0, stream>>>(
			thrust::raw_pointer_cast(d_column_sizes.data()), 
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data())
		);
		rows = in_rows;
	}

	void check_if_matrix_reduced(thrust::device_vector<int32_t>& rank_search_flags, cudaStream_t stream) {
		check_if_matrix_reduced_raw<<<BLOCKS, THREADS, 0, stream>>>(
			thrust::raw_pointer_cast(rank_search_flags.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data())
		);
	}

	void find_subtraction_pairs(
		thrust::device_vector<int32_t>& d_nnz_estimation,
		thrust::device_vector<int32_t>& d_pairs_for_subtractions,
		cudaStream_t stream) {
		find_subtraction_pairs_raw<<<BLOCKS, THREADS, 0, stream>>>(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data())
		);
	}

	// TODO: add squash method to remove all garbage data in d_rows_indicies

	void perform_subtraction(CSRMatrix& output, const thrust::device_vector<int32_t>& d_pairs_for_subtractions, cudaStream_t stream) const {
		perform_subtractions<<<BLOCKS, 1, 0, stream>>>(
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),

			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),

			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data())
		);
	}

	void update_columns_offsets(thrust::device_vector<int32_t>& d_nnz_estimation, CSRMatrix& output) const {
		thrust::host_vector<int32_t> nnz_estimation = d_nnz_estimation;
		thrust::host_vector<int32_t> column_sizes = d_column_sizes;
		thrust::host_vector<int32_t> new_columns_offsets(output.d_column_sizes.size() + 1, (int32_t)0);

		for (size_t i = 1; i < output.d_column_sizes.size() + 1; ++i) {
			const int32_t estimation = (nnz_estimation[i - 1] == COLUMN_STAYS_FIXED) ? column_sizes[i - 1] : nnz_estimation[i - 1];
			new_columns_offsets[i] = new_columns_offsets[i - 1] + estimation;
		}

		output.d_columns_offsets = new_columns_offsets;
		output.d_rows_indicies.assign(new_columns_offsets[output.d_column_sizes.size()], INVALID_VALUE);
		output.d_column_sizes.assign(output.d_column_sizes.size(), 0);
	}

	void move_fixed_columns(CSRMatrix& output, thrust::device_vector<int32_t>& d_nnz_estimation, cudaStream_t stream) const {
		move_fixed_columns_raw<<<BLOCKS, THREADS, 0, stream>>>(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),

			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			
			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data()),
			d_column_sizes.size()
		);
	}

	void squash_memory(CSRMatrix& output, cudaStream_t stream) {
		// Count nnz elements in csr
		size_t total_elements_needed = thrust::reduce(
			thrust::device, 
			d_column_sizes.begin(), 
			d_column_sizes.end(),
			0
		);
		output.d_rows_indicies.resize(total_elements_needed);

		// Find new column_offsets
		thrust::host_vector<int32_t> column_sizes = d_column_sizes;
		thrust::host_vector<int32_t> new_columns_offsets(output.d_column_sizes.size() + 1, (int32_t)0);

		for (size_t i = 1; i < output.d_column_sizes.size() + 1; ++i) {
			const int32_t estimation = column_sizes[i - 1];
			new_columns_offsets[i] = new_columns_offsets[i - 1] + estimation;
		}

		output.d_columns_offsets = new_columns_offsets;
		
		// Move all columns together
		move_fixed_columns_raw<<<BLOCKS, THREADS, 0, stream>>>(
			nullptr,

			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),

			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data()),
			d_column_sizes.size()
		);

		d_column_sizes = output.d_column_sizes;
		d_rows_indicies = output.d_rows_indicies;
		d_columns_offsets = output.d_columns_offsets;
		d_max_row_indexes = output.d_max_row_indexes;
	}

	int32_t find_rank() {
		// If matrix is not fully reduced, the result may be incorrect
		return thrust::transform_reduce(d_column_sizes.begin(), d_column_sizes.end(),
			is_positive<int32_t>(),
			0,
			thrust::plus<int32_t>());
	}

	void print() {
		thrust::host_vector<int32_t> column_sizes = d_column_sizes;
		thrust::host_vector<int32_t> rows_indicies = d_rows_indicies;
		thrust::device_vector<int32_t> columns_offsets = d_columns_offsets;

		for (int32_t column_id = 0; column_id < column_sizes.size(); ++column_id) {
			std::cout << "Column " << column_id << " (column size " << column_sizes[column_id] << "): ";
			for (int32_t element_id = columns_offsets[column_id]; element_id < columns_offsets[column_id + 1]; ++element_id) {
				std::cout << rows_indicies[element_id] << " ";
			}
			std::cout << "\n";
		}
	}

	void log_memory_consumption() {
		std::cout << "Memory consumption (int32_t's): " << \
			"c_off: " << d_columns_offsets.capacity() << \
			" c_sizes: " << d_column_sizes.capacity() << \
			" max_rows: " << d_max_row_indexes.capacity() << \
			" row_inds: " << d_rows_indicies.capacity();
		std::cout << "\n";
		std::cout << "row_inds size: " << d_rows_indicies.size();
		std::cout << "\n";
	}
};

extern "C" int32_t find_rank_raw(const int32_t* column_offsets, const uint32_t column_offsets_len, const int32_t* rows_indicies, const uint32_t nnz, const int32_t columns, const int32_t rows, const int32_t max_attempts) {
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	
	CSRMatrix buffers[] = {
		CSRMatrix(column_offsets, column_offsets_len, rows_indicies, nnz, columns, rows, stream1),
		CSRMatrix(columns, rows)
	};
	uint32_t active_buffer_index = 0;

#ifdef DEBUG_PRINT
	buffers[active_buffer_index].print();
#endif
	
	thrust::device_vector<int32_t> rank_search_flags(RANK_SEARCH_FLAGS_SIZE, 0);
	// Structure of rank_search_flags:
	// 0) is matrix reduced?

	thrust::device_vector<int32_t> d_pairs_for_subtractions(PAIRS_PER_ROUND * 2, INVALID_PAIR_VALUE);
	thrust::device_vector<int32_t> d_nnz_estimation(columns, COLUMN_STAYS_FIXED);
	cudaCheckError("Buffer initialisation");

	for (int32_t attempt = 0; (attempt < max_attempts) && (rank_search_flags[0] == 0); ++attempt) {
		d_pairs_for_subtractions.assign(PAIRS_PER_ROUND * 2, INVALID_PAIR_VALUE);
		buffers[active_buffer_index].find_subtraction_pairs(d_nnz_estimation, d_pairs_for_subtractions, stream1);
		
#ifdef DEBUG_PRINT
		printf("\n");
#endif
		
		buffers[active_buffer_index].update_columns_offsets(d_nnz_estimation, buffers[1 - active_buffer_index]);
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);
		buffers[active_buffer_index].perform_subtraction(buffers[1 - active_buffer_index], d_pairs_for_subtractions, stream2);
		buffers[active_buffer_index].move_fixed_columns(buffers[1 - active_buffer_index], d_nnz_estimation, stream1);
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);

		active_buffer_index = 1 - active_buffer_index;
		rank_search_flags[0] = 1;
		buffers[active_buffer_index].check_if_matrix_reduced(rank_search_flags, stream1);
		if (attempt % SQUASHING_DELAY == (SQUASHING_DELAY - 1)) {
			buffers[active_buffer_index].squash_memory(buffers[1 - active_buffer_index], stream1);
			#ifdef LOG_MEMORY_CONSUMPTION
				buffers[active_buffer_index].log_memory_consumption();
			#endif
		}
		cudaCheckError("Matrix reduction check");

#ifdef DEBUG_PRINT
		std::flush(std::cout);
		std::cout << "Attempt " << attempt << ", rank " << buffers[active_buffer_index].find_rank() << "\n";
		buffers[active_buffer_index].print();
#endif
	}

	std::flush(std::cout);

	return buffers[active_buffer_index].find_rank();
}
