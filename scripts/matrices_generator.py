from scipy.sparse import save_npz, coo_matrix, rand
from tqdm import tqdm
import argparse
import numpy as np
import os


# Try to create directory to store matrices
def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        return


# Generate sparce random matrix
def pseudo_random_sparce(m, n, density=0.01, format='coo'):
    # Copy-paste from scipy.sparse.rand with several exceptions
    tp = np.intc
    k = int(round(density * m * n))

    ind = np.random.choice(m * n, size=k, replace=True)

    j = np.floor(ind * 1. / m).astype(tp, copy=False)
    i = (ind - j * m).astype(tp, copy=False)

    return coo_matrix((np.ones(k), (i, j)), shape=(m, n)).asformat(format,
                                                             copy=False)


def main():  
    # Parse arguements
    parser = argparse.ArgumentParser(
                prog='ParallelRankDataParser',
                description='This program creates data needed to build plots')
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    # Data generation parametres
    sides = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    repititions = 5
    const_width = 100
    
    if args.test:
        sides = sides[:2]
        
    main_dir = "parallelrank_data"
    create_dir(main_dir)

    print("Generating matrices with different sides in directory \"{}\":".format(main_dir))
    for side in sides:
        # Generate matrix with changing and constant width
        print("Generating matrices with side", side)
        for nnz, shape in tqdm([
                (const_width * const_width, (const_width, side)), 
                (1 * side, (side, side)), 
                (10 * side, (side, side)), 
                (100 * 100, (side, side))
        ]):
            width, height = shape

            create_dir("{}/shape_{}_{}".format(main_dir, width, height))
            
            # Use several repititions to get more precise results from average data
            for i in range(repititions):
                density = nnz / (width * height)
                
                if side > 10_000:
                    matrix = pseudo_random_sparce(width, height, density, 'csr')
                else:
                    matrix = rand(width, height, density=density, format='csr', dtype=np.int8)
                matrix.data[:] = 1
                
                save_npz("{}/shape_{}_{}/matrix_nnz_{}_iter_{}.npz".format(main_dir, width, height, nnz, i), matrix)


if __name__ == "__main__":
    main()
