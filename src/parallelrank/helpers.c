#include <stdlib.h>

extern int findRankRaw(float* matrix, size_t rows, size_t columns);

// example for export in python
int add(int a, int b) {
	return a + b;
}

int findRank(float* matrix, size_t rows, size_t columns) {
	if (matrix == NULL) {
		return -1;
	}

	return findRankRaw(matrix, rows, columns);
}
