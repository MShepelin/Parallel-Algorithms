#include <iostream>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

#define CPU_PAIRS_PER_ROUND 1024
#define CPU_SQUASHING_DELAY 1024

#define CPU_COLUMN_STAYS_FIXED -1
#define CPU_INVALID_PAIR_VALUE -1
#define CPU_INVALID_VALUE -1

struct ThreadInfo {
	uint32_t thread_id;
	uint32_t thread_num;
};

__inline__ int32_t get_mask_from_bool(int32_t a) {
	return ~(a - 1);
}

// TODO: change type in subtraction_pairs for uint32_t
void cpu_perform_subtractions(
	int32_t* subtraction_pairs,
	const int32_t* input_columns_offsets,
	const int32_t* input_column_sizes,
	const int32_t* input_rows_indicies,
	const int32_t* output_columns_offsets,
	int32_t* output_column_sizes,
	int32_t* output_rows_indicies,
	int32_t* output_max_row_indexes,
	ThreadInfo t_info
) {
	// Assumes subtraction_pairs has size (CPU_PAIRS_PER_ROUND * 2)
	for (
		int32_t pair_id = t_info.thread_id;
		pair_id < CPU_PAIRS_PER_ROUND; 
		pair_id += t_info.thread_num
	) {
		const int32_t column_from = subtraction_pairs[pair_id * 2];
		const int32_t column_subtraction = subtraction_pairs[pair_id * 2 + 1];

		if (column_from == CPU_INVALID_PAIR_VALUE || column_subtraction == CPU_INVALID_PAIR_VALUE) {
			continue;
		}

		subtraction_pairs[pair_id * 2] = subtraction_pairs[pair_id * 2 + 1] = CPU_INVALID_PAIR_VALUE;

		uint32_t id_to_put = output_columns_offsets[column_from];
		uint32_t left_column_id = input_columns_offsets[column_from];
		const uint32_t left_column_id_limit = left_column_id + input_column_sizes[column_from];
		uint32_t right_column_id = input_columns_offsets[column_subtraction];
		const uint32_t right_column_id_limit = right_column_id + 
			input_column_sizes[column_subtraction];

		output_max_row_indexes[column_from] = 0;
		output_column_sizes[column_from] = 0;
		int32_t column_from_size = 0;
		while (
			left_column_id < left_column_id_limit || 
			right_column_id < right_column_id_limit
		) {
			int32_t left_low = (left_column_id < left_column_id_limit) ? 
				input_rows_indicies[left_column_id] : INT32_MAX;
			int32_t right_low = (right_column_id < right_column_id_limit) ? 
				input_rows_indicies[right_column_id] : INT32_MAX;

			#ifdef DEBUG_PRINT
			if (left_low == CPU_INVALID_VALUE || right_low == CPU_INVALID_VALUE) {
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

void cpu_move_fixed_columns_raw(
	const int32_t* nnz_estimation,
	const int32_t* input_column_sizes,
	const int32_t* input_rows_indicies,
	const int32_t* input_columns_offsets,
	const int32_t* input_max_row_indexes,
	int32_t* output_column_sizes,
	int32_t* output_rows_indicies,
	const int32_t* output_columns_offsets,
	int32_t* output_max_row_indexes,
	uint32_t columns,
	ThreadInfo t_info
) {
	for (
		int32_t column_id = t_info.thread_id;
		column_id < columns;
		column_id += t_info.thread_num
	) {
		if (nnz_estimation != nullptr && nnz_estimation[column_id] != CPU_COLUMN_STAYS_FIXED) {
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
			output_rows_indicies[output_row_id + id_delta] = 
				input_rows_indicies[input_row_id + id_delta];
		}
	}
}

void cpu_find_subtraction_pairs_raw(
	int32_t* nnz_estimation, 
	int32_t* subtraction_pairs, 
	const int32_t* column_sizes, 
	int32_t columns, 
	const int32_t* columns_offset, 
	const int32_t* rows_indices, 
	const int32_t* max_row_indexes,
	ThreadInfo t_info
) {
	int32_t new_subtraction_id = 0;
	uint32_t max_subtractions = CPU_PAIRS_PER_ROUND / t_info.thread_num;
	uint32_t offset = t_info.thread_id * max_subtractions;

	for (size_t column_id = t_info.thread_id; column_id < columns; column_id += t_info.thread_num) {
		const int32_t column_size = column_sizes[column_id];

		if (column_size == 0) {
			nnz_estimation[column_id] = CPU_COLUMN_STAYS_FIXED;
		}
		else {
			const int32_t max_row_index = max_row_indexes[column_id];
			int32_t pivot = CPU_INVALID_VALUE;
			int32_t nnz = INT32_MAX;

			// Find column with minimum number of nnz for each thread
			for (
				size_t column_compare_id = 0;
				column_compare_id < columns;
				++column_compare_id
			) {
				// Here we find the best pivot by comparison
				// Comparison is presented in boolean form (without if-statements)
				const int32_t column_compare_size = column_sizes[column_compare_id];
				const int32_t comp_value = (
					max_row_indexes[column_compare_id] == max_row_index &&
					column_compare_size < nnz
				);
				const int32_t comp_value_mask = get_mask_from_bool(comp_value);

				nnz = ((~comp_value_mask) & nnz) |
					(comp_value_mask & column_compare_size);
				pivot = ((~comp_value_mask) & pivot) |
					(comp_value_mask & column_compare_id);
			}

			if (pivot != CPU_INVALID_VALUE &&
				pivot != column_id &&
				new_subtraction_id < max_subtractions
			) {
				nnz_estimation[column_id] = column_size + nnz - 2;
				subtraction_pairs[(offset + new_subtraction_id) << 1] = column_id;
				subtraction_pairs[((offset + new_subtraction_id) << 1) + 1] = pivot;
				// subtraction pair means columns[column_id] -= columns[best_pivot]

				++new_subtraction_id;
			}
			else {
				nnz_estimation[column_id] = CPU_COLUMN_STAYS_FIXED;
			}
		}
	}
}

void cpu_check_if_matrix_reduced_raw(
	std::atomic_int& is_reduced,
	int32_t* column_sizes,
	uint32_t columns,
	const int32_t* column_offsets,
	int32_t* max_row_indexes,
	const int32_t* rows_indices,
	ThreadInfo t_info
) {
	for (size_t column_left = t_info.thread_id; column_left < columns; column_left += t_info.thread_num) {
		if (column_sizes[column_left] <= 0) {
			continue;
		}

		const int32_t left_max_row_index = max_row_indexes[column_left];

		for (
			size_t column_right = column_left + 1; 
			column_right < columns; 
			++column_right
		) {
			if (left_max_row_index == max_row_indexes[column_right]) {
				is_reduced = 0;
			}
		}
	}
}

void cpu_fill_column_sizes(
	int32_t* column_sizes, 
	uint32_t columns, 
	const int32_t* columns_offsets, 
	int32_t* max_row_indexes, 
	const int32_t* rows_indicies,
	ThreadInfo t_info
) {
	// Assumes columns_offsets has size of (columns + 1)
	// Assumes rows_indicies doesn't have fake values
	for (uint32_t i = t_info.thread_id; i < columns; i += t_info.thread_num) {
		const size_t next_column_offset = columns_offsets[i + 1];
		const size_t column_size = next_column_offset - columns_offsets[i];
		column_sizes[i] = column_size;
		
		if (column_size > 0 && next_column_offset >= 1) {
			max_row_indexes[i] = rows_indicies[next_column_offset - 1];
		}
		else {
			max_row_indexes[i] = CPU_INVALID_VALUE;
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

void cpu_update_columns_offsets_raw(
	const int32_t* nnz_estimation,
	const int32_t* input_column_sizes,
	int32_t* output_columns_offsets,
	uint32_t columns
) {
	// Assumes that this function is run with 1 block
	for (uint32_t i = 1; i < columns + 1; ++i) {
		const int32_t estimation = (nnz_estimation[i - 1] == CPU_COLUMN_STAYS_FIXED) ? 
			input_column_sizes[i - 1] : nnz_estimation[i - 1];
		output_columns_offsets[i] = output_columns_offsets[i - 1] + estimation;
	}
}

void cpu_memory_squash_raw(
	const int32_t* input_column_sizes,
	int32_t* output_columns_offsets,
	uint32_t columns
) {
	// Assumes that this function is run with 1 block
	for (size_t i = 1; i < columns + 1; ++i) {
		const int32_t estimation = input_column_sizes[i - 1];
		output_columns_offsets[i] = output_columns_offsets[i - 1] + estimation;
	}
}

struct CSRMatrixCPU {
private:
	int32_t rows;
	thrust::host_vector<int32_t> d_columns_offsets;
	thrust::host_vector<int32_t> d_rows_indicies;
	// Number of real elements in column,
	// is <= (difference in d_columns_offsets neighbour elements)
	thrust::host_vector<int32_t> d_column_sizes; 
	thrust::host_vector<int32_t> d_max_row_indexes;

public:
	CSRMatrixCPU() = delete;

	CSRMatrixCPU(const int32_t in_columns, const int32_t in_rows) {
		d_column_sizes.assign(in_columns, 0);
		d_max_row_indexes.assign(in_columns, 0);
		// We put invalid size values in d_columns_offsets
		d_columns_offsets.assign(in_columns + 1, CPU_INVALID_VALUE);
		// d_rows_indicies stays empty
		
		// TODO: check that d_columns_offsets really has size (columns + 1)
		rows = in_rows;
	}

	CSRMatrixCPU(
		const int32_t* column_offsets,
		const uint32_t column_offsets_len,
		const int32_t* rows_indicies,
		const uint32_t nnz,
		const int32_t in_columns,
		const int32_t in_rows
	) {
		d_column_sizes.assign(in_columns, 0);
		d_max_row_indexes.assign(in_columns, 0);
		d_columns_offsets.assign(column_offsets, column_offsets + column_offsets_len);
		d_rows_indicies.assign(rows_indicies, rows_indicies + nnz);

		cpu_fill_column_sizes(
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			{ 0, 1 }
		);
		
		rows = in_rows;
	}

	void check_if_matrix_reduced(
		std::atomic_int& is_reduced
	) {
		cpu_check_if_matrix_reduced_raw(
			is_reduced,
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			{ 0, 1 }
		);
	}

	void find_subtraction_pairs(
		thrust::host_vector<int32_t>& d_nnz_estimation,
		thrust::host_vector<int32_t>& d_pairs_for_subtractions
	) {
		cpu_find_subtraction_pairs_raw(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			d_column_sizes.size(),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),
			{ 0, 1 }
		);
	}

	void perform_subtraction(
		CSRMatrixCPU& output, 
		thrust::host_vector<int32_t>& d_pairs_for_subtractions
	) const {
		cpu_perform_subtractions(
			thrust::raw_pointer_cast(d_pairs_for_subtractions.data()),

			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),

			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data()),

			{ 0, 1 }
		);
	}

	void update_columns_offsets(
		thrust::host_vector<int32_t>& d_nnz_estimation, 
		CSRMatrixCPU& output
	) const {
		uint32_t columns = d_column_sizes.size();
		output.d_columns_offsets.assign(columns + 1, 0);

		cpu_update_columns_offsets_raw(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),
			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			columns
		);

		output.d_rows_indicies.assign(output.d_columns_offsets[columns], CPU_INVALID_VALUE);
		output.d_column_sizes.assign(columns, 0);
	}

	void move_fixed_columns(
		CSRMatrixCPU& output, 
		thrust::host_vector<int32_t>& d_nnz_estimation
	) const {
		cpu_move_fixed_columns_raw(
			thrust::raw_pointer_cast(d_nnz_estimation.data()),

			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),

			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data()),
			d_column_sizes.size(),

			{ 0, 1 }
		);
	}

	void squash_memory(CSRMatrixCPU& output) {
		// Count nnz elements in csr
		size_t total_elements_needed = 0;
		for (size_t i = 0; i < d_column_sizes.size(); ++i) {
			total_elements_needed += d_column_sizes[i];
		}
		output.d_rows_indicies.resize(total_elements_needed);

		// Find new column_offsets
		uint32_t columns = d_column_sizes.size();

		cpu_memory_squash_raw(
			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			columns
		);
		
		cpu_move_fixed_columns_raw(
			nullptr,

			thrust::raw_pointer_cast(d_column_sizes.data()),
			thrust::raw_pointer_cast(d_rows_indicies.data()),
			thrust::raw_pointer_cast(d_columns_offsets.data()),
			thrust::raw_pointer_cast(d_max_row_indexes.data()),

			thrust::raw_pointer_cast(output.d_column_sizes.data()),
			thrust::raw_pointer_cast(output.d_rows_indicies.data()),
			thrust::raw_pointer_cast(output.d_columns_offsets.data()),
			thrust::raw_pointer_cast(output.d_max_row_indexes.data()),
			d_column_sizes.size(),

			{ 0, 1 }
		);

		d_column_sizes = output.d_column_sizes;
		d_rows_indicies = output.d_rows_indicies;
		d_columns_offsets = output.d_columns_offsets;
		d_max_row_indexes = output.d_max_row_indexes;
	}

	int32_t find_rank() {
		// If matrix is not fully reduced, the result may be incorrect
		int32_t rank = 0;
		for (int i = 0; i < d_column_sizes.size(); ++i) {
			if (d_column_sizes[i] > 0) {
				++rank;
			}
		}
		return rank;
	}

	void print() {
		thrust::host_vector<int32_t> column_sizes = d_column_sizes;
		thrust::host_vector<int32_t> rows_indicies = d_rows_indicies;
		thrust::host_vector<int32_t> columns_offsets = d_columns_offsets;

		for (int32_t column_id = 0; column_id < column_sizes.size(); ++column_id) {
			std::cout << "Column " << column_id;
			std::cout << " (column size " << column_sizes[column_id] << "): ";
			for (
				int32_t element_id = columns_offsets[column_id]; 
				element_id < columns_offsets[column_id + 1]; 
				++element_id
			) {
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

	size_t get_memory_consumption() {
		return d_columns_offsets.capacity() +
			d_column_sizes.capacity() +
			d_max_row_indexes.capacity() +
			d_rows_indicies.capacity();
	}
};

extern "C" int32_t find_rank_cpu(
	const int32_t* column_offsets, 
	const uint32_t column_offsets_len, 
	const int32_t* rows_indicies, 
	const uint32_t nnz, 
	const int32_t columns, 
	const int32_t rows, 
	const int32_t max_attempts
) {
	CSRMatrixCPU buffers[] = {
		CSRMatrixCPU(column_offsets, column_offsets_len, rows_indicies, nnz, columns, rows),
		CSRMatrixCPU(columns, rows)
	};
	uint32_t active_buffer_index = 0;

	#ifdef DEBUG_PRINT
	buffers[active_buffer_index].print();
	#endif
	
	std::atomic_int is_reduced;
	is_reduced = 0;
	// Structure of rank_search_flags:
	// 0) is matrix reduced?

	thrust::host_vector<int32_t> d_pairs_for_subtractions(
		CPU_PAIRS_PER_ROUND * 2, 
		CPU_INVALID_PAIR_VALUE
	);
	thrust::host_vector<int32_t> d_nnz_estimation(columns, CPU_COLUMN_STAYS_FIXED);

	for (int32_t attempt = 0; (attempt < max_attempts) && (is_reduced == 0); ++attempt) {
		//d_pairs_for_subtractions.assign(CPU_PAIRS_PER_ROUND * 2, CPU_INVALID_PAIR_VALUE);
		buffers[active_buffer_index].find_subtraction_pairs(
			d_nnz_estimation,
			d_pairs_for_subtractions
		);

		#ifdef DEBUG_PRINT
		printf("\npairs ");
		for (const auto& el : d_pairs_for_subtractions) {
			std::cout << el << " ";
		}
		printf("\n");
		#endif
		
		buffers[active_buffer_index].update_columns_offsets(
			d_nnz_estimation, 
			buffers[1 - active_buffer_index]
		);

		buffers[active_buffer_index].perform_subtraction(
			buffers[1 - active_buffer_index], 
			d_pairs_for_subtractions
		);
		buffers[active_buffer_index].move_fixed_columns(
			buffers[1 - active_buffer_index], 
			d_nnz_estimation
		);

		active_buffer_index = 1 - active_buffer_index;
		is_reduced = 1;
		buffers[active_buffer_index].check_if_matrix_reduced(is_reduced);
		if (attempt % CPU_SQUASHING_DELAY == (CPU_SQUASHING_DELAY - 1)) {
			buffers[active_buffer_index].squash_memory(buffers[1 - active_buffer_index]);
			#ifdef LOG_MEMORY_CONSUMPTION
				buffers[active_buffer_index].log_memory_consumption();
			#endif
		}

		#ifdef DEBUG_PRINT
		std::flush(std::cout);
		std::cout << "Attempt " << attempt;
		std::cout << ", rank " << buffers[active_buffer_index].find_rank() << "\n";
		buffers[active_buffer_index].print();
		#endif
	}

	#ifdef DEBUG_PRINT
	std::flush(std::cout);
	#endif
	
	return buffers[active_buffer_index].find_rank();
}
