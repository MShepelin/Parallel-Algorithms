#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDA-By-Example/common/book.h"
#include "CUDA-By-Example/common/cpu_bitmap.h"

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
