#include <algorithm>
#include <cmath>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

class GaussRank {
private:
    size_t rows;
    size_t columns_div_32;
    std::vector<bool> pivots;

    void substract_collumn(std::vector<std::vector<uint32_t>>& matrix, size_t from, size_t which) {
        for (int i = 0; i < columns_div_32; ++i) {
            matrix[from][i] = matrix[from][i] ^ matrix[which][i];
        }
    }

    bool is_nnz(std::vector<std::vector<uint32_t>>& matrix, size_t row, size_t column) {
        return matrix[row][column >> 5] & ((uint32_t)1 << (column % 32));
    }

public:
    int32_t find_rank(std::vector<std::vector<uint32_t>>& matrix) {
        int32_t rank = 0;

        for (size_t column_id = 0; column_id < columns_div_32 * 32; ++column_id) {
            // Find pivot
            size_t pivot = 0;
            bool found_pivot = false;
            for (size_t row_id = 0; row_id < rows; ++row_id) {
                if (pivots[row_id]) {
                    continue;
                }

                if (is_nnz(matrix, row_id, column_id)) {
                    pivot = row_id;
                    found_pivot = true;
                    break;
                }
            }

            if (!found_pivot) {
                continue;
            }

            // Subtract pivot
            pivots[pivot] = true;
            ++rank;
            for (size_t row_id = 0; row_id < rows; ++row_id) {
                if (!pivots[row_id] && is_nnz(matrix, row_id, column_id)) {
                    substract_collumn(matrix, row_id, pivot);
                }
            }
        }

        return rank;
    }

    GaussRank(size_t in_rows, size_t in_columns_div_32) : 
        rows(in_rows),
        columns_div_32(in_columns_div_32),
        pivots(in_rows, false)
    { }
};

extern "C" int32_t find_rank_gauss(
	const int32_t* column_offsets, 
	const uint32_t column_offsets_len, 
	const int32_t* rows_indicies, 
	const uint32_t nnz, 
	const int32_t columns, 
	const int32_t rows,
    int32_t* memory_consumption
) {
    size_t columns_div_32 = (columns + 31) >> 5;
    GaussRank calculator(rows, columns_div_32);
    std::vector<std::vector<uint32_t>> matrix(rows, std::vector<uint32_t>(columns_div_32, 0));

    for (size_t column_offset = 1; column_offset < column_offsets_len; ++column_offset) {
        for (
            size_t row_id = column_offsets[column_offset - 1]; 
            row_id < column_offsets[column_offset];
            ++row_id
        ) {
            size_t row = rows_indicies[row_id];
            size_t column = column_offset - 1;
            matrix[row][column >> 5] |= ((uint32_t)1 << (column % 32));
        }
    }

    *memory_consumption = matrix.capacity() * matrix[0].capacity() * 4 + columns / 8;
    return calculator.find_rank(matrix);
}
