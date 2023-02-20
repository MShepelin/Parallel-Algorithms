#include <stdlib.h>
#include <stdint.h>

extern void read_CSR(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns, int32_t rows);

void read_CSR_matrix(int32_t* column_offsets, uint32_t column_offsets_len, int32_t* rows_indicies, uint32_t nnz, int32_t columns, int32_t rows) {
	read_CSR(column_offsets, column_offsets_len, rows_indicies, nnz, columns, rows);
}
