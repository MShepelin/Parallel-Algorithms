#include <stdlib.h>
#include <stdint.h>

extern int32_t find_rank_raw(
	const int32_t* column_offsets, 
	const uint32_t column_offsets_len, 
	const int32_t* rows_indicies, 
	const uint32_t nnz, 
	const int32_t columns, 
	const int32_t rows, 
	const int32_t max_attempts
);

int32_t find_rank(
	const int32_t* column_offsets, 
	const uint32_t column_offsets_len, 
	const int32_t* rows_indicies, 
	const uint32_t nnz, 
	const int32_t columns, 
	const int32_t rows, 
	const int32_t max_attempts
) {
	find_rank_raw(
		column_offsets, 
		column_offsets_len, 
		rows_indicies, 
		nnz, 
		columns, 
		rows, 
		max_attempts
	);
}
