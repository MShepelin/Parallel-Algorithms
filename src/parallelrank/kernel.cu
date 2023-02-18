#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDA-By-Example/common/book.h"
#include "CUDA-By-Example/common/cpu_bitmap.h"
#include <thrust/device_vector.h>

#define GRID_DIM 1000

__global__ void gaussStep(float* matrix, const size_t rows, const size_t columns, const size_t activeRowId, bool* foundRowToSwap) {
	if (!(*foundRowToSwap)) {
		return;
	} 
	
	for (size_t currentRowId = blockIdx.x; currentRowId < rows; currentRowId += gridDim.x) {
		if (currentRowId == activeRowId) {
			continue;
		}

		float multiplier = matrix[currentRowId * columns + activeRowId] / matrix[activeRowId * columns + activeRowId];

		// Subtract active row from current row
		for (size_t currentRowElementId = activeRowId; currentRowElementId < columns; ++currentRowElementId) {
			matrix[currentRowId * columns + currentRowElementId] -= multiplier * matrix[activeRowId * columns + currentRowElementId];
		}
	}
}

__global__ void swapRows(float* matrix, const size_t columns, const size_t activeRowId, const size_t* rowToSwap, bool* foundRowToSwap) {
	if (!(*foundRowToSwap)) {
		return;
	}
	for (size_t i = blockIdx.x; i < columns; i += gridDim.x) {
		const float swappedElement = matrix[(*rowToSwap) * columns + i];
		matrix[(*rowToSwap) * columns + i] = matrix[activeRowId * columns + i];
		matrix[activeRowId * columns + i] = swappedElement;
	}
}

__global__ void findRowToSwap(
	float* matrix, 
	const size_t rows,
	const size_t columns,
	const size_t activeRowId, 
	size_t* rowToSwap,
	size_t* rankDecrease,
	bool* foundRowToSwap)
{
	*foundRowToSwap = false;
	while (activeRowId + *rankDecrease < columns) {
		size_t activeColumId = activeRowId + *rankDecrease;

		// Scan the active column
		// TODO: may be add reduction to avoid scanning a long column
		for (size_t nonzeroRowElement = activeRowId; nonzeroRowElement < rows; ++nonzeroRowElement) {
			if (matrix[nonzeroRowElement * columns + activeRowId] != 0) {
				*foundRowToSwap = true;
				*rowToSwap = activeRowId;
				return;
			}
		}

		// Didn't find row to swap
		// Cannot do row subtraction
		++(*rankDecrease);
	}
}

float* inputMatrix(size_t* rows, size_t* columns) {
	printf("Enter number of rows and columns:\n");
	int readArguementsNum = scanf("%llu %llu", rows, columns);
	if (readArguementsNum != 2 || (*rows) <= 0 || (*columns) <= 0) {
		printf("Wrong rows and columns input\n");
		return nullptr;
	}

	size_t matrixMallocSize = sizeof(float) * (*rows) * (*columns);
	float* matrix = (float*) malloc(matrixMallocSize);
	if (matrix == nullptr) {
		printf("Matrix memory allocation error\n");
		return nullptr;
	}

	for (size_t i = 0; i < (*rows); ++i) {
		for (size_t j = 0; j < (*columns); ++j) {
			readArguementsNum = scanf("%f", &matrix[(*columns) * i + j]);
			if (readArguementsNum != 1) {
				printf("Wrong input in matrix\n");
				free(matrix);
				return nullptr;
			}
		}
	}

	return matrix;
}

extern "C" int findRankRaw(float* matrix, size_t rows, size_t columns) {
	// Assume that matrix is non-null and points to data of size rows*columns

	size_t matrixMallocSize = sizeof(float) * rows * columns;
	size_t minSide = min(rows, columns);
	size_t rankDecrease = 0;

	size_t* devRankDecrease = nullptr;
	bool* devFoundRowToSwap = nullptr;
	float* devMatrix = nullptr;
	size_t* devRowToSwap = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&devMatrix, matrixMallocSize));
	HANDLE_ERROR(cudaMalloc((void**)&devRankDecrease, sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void**)&devFoundRowToSwap, sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void**)&devRowToSwap, sizeof(size_t)));
	HANDLE_ERROR(cudaMemcpy(devMatrix,
		matrix,
		matrixMallocSize,
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(devRankDecrease,
		&rankDecrease,
		sizeof(size_t),
		cudaMemcpyHostToDevice));

	for (size_t activeRowId = 0; activeRowId < minSide; ++activeRowId) {
		// Find row with nonzero element on active column
		findRowToSwap<<<1, 1>>>(
			devMatrix, 
			rows, 
			columns, 
			activeRowId,
			devRowToSwap, 
			devRankDecrease,
			devFoundRowToSwap);
		// Swap found row and active row
		swapRows<<<min((size_t)GRID_DIM, columns), 1>>>(
			devMatrix,
			columns,
			activeRowId,
			devRowToSwap,
			devFoundRowToSwap);
		// Subtract active row from other rows
		// TODO: parallelize gaussStep using grid
		gaussStep<<<min((size_t)GRID_DIM, rows), 1>>>(
			devMatrix, 
			rows, 
			columns, 
			activeRowId, 
			devFoundRowToSwap);
	}
	
	HANDLE_ERROR(cudaMemcpy(matrix,
		devMatrix,
		matrixMallocSize,
		cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&rankDecrease,
		devRankDecrease,
		sizeof(size_t),
		cudaMemcpyDeviceToHost));

	cudaFree(devMatrix);
	cudaFree(devRankDecrease);
	cudaFree(devFoundRowToSwap);
	cudaFree(devRowToSwap);
	return minSide - rankDecrease;
}

#define RANK_SEARCH_FLAGS_SIZE 1

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
};

extern "C" void read_CSR(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns, int32_t rows) {
	CSRMatrix buffer_1(column_offsets, column_offsets_len, rows_indicies, nnz, columns);
	CSRMatrix buffer_2;
	bool is_first_buffer_garbage = false;
	cudaCheckError("Buffer initialisation");

	thrust::device_vector<int32_t> rank_search_flags(RANK_SEARCH_FLAGS_SIZE, false);
	// Structure of rank_search_flags:
	// 0) is matrix reduced?

	// Do while not reduced:
	//while (!rank_search_flags[0]) { // TODO: figure out a better way to check boolean
		// Compute

		if (is_first_buffer_garbage) {
			buffer_2.check_if_matrix_reduced(rank_search_flags);
		}
		else {
			buffer_1.check_if_matrix_reduced(rank_search_flags);
		}

		cudaCheckError("Matrix reduction check");
	//}
}
