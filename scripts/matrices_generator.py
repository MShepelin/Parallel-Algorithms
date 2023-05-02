from helpers import create_dir, get_sides, REPETITIONS, CONST_WIDTH, DATA_DIR
from scipy.sparse import save_npz, coo_matrix, rand
from tqdm import tqdm
import argparse
import numpy as np


# Generate sparce random matrix
def pseudo_random_sparce(m, n, density=0.01, format='coo'):
    # Copy-paste from scipy.sparse.rand with several exceptions
    tp = np.intc
    k = int(round(density * m * n))

    ind = np.random.choice(m * n, size=k, replace=True)

    j = np.floor(ind * 1. / m).astype(tp, copy=False)
    i = (ind - j * m).astype(tp, copy=False)

    return coo_matrix((np.ones(k), (i, j)), shape=(m, n)).asformat(
        format,
        copy=False
    )


def main():  
    # Parse arguements
    parser = argparse.ArgumentParser(
                prog='ParallelRankDataGenerator',
                description='This program creates data needed to build plots')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    # Data generation parametres
    sides = get_sides(args.test)

    main_dir = DATA_DIR
    create_dir(main_dir)

    print("Generating matrices with different sides in directory \"{}\":".format(main_dir))
    for side in sides:
        # Generate matrix with changing and constant width
        print("Generating matrices with side", side)
        for nnz, shape in tqdm([
                (CONST_WIDTH * CONST_WIDTH, (CONST_WIDTH, side)), 
                (1 * side, (side, side)), 
                (10 * side, (side, side)), 
                (CONST_WIDTH * CONST_WIDTH, (side, side))
        ]):
            width, height = shape

            create_dir("{}/shape_{}x{}".format(main_dir, width, height))
            
            # Use several repetitions to get more precise results from average data
            for i in range(REPETITIONS):
                density = nnz / (width * height)
                
                if side > 10_000:
                    matrix = pseudo_random_sparce(width, height, density, 'csr')
                else:
                    matrix = rand(width, height, density=density, format='csr', dtype=np.int8)
                matrix.data[:] = 1
                
                save_npz("{}/shape_{}x{}/matrix_nnz_{}_iter_{}.npz".format(main_dir, width, height, nnz, i), matrix)


if __name__ == "__main__":
    main()
