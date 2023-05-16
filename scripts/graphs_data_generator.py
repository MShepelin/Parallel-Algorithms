from helpers import create_dir, get_sides, REPETITIONS, CONST_WIDTH, DATA_DIR
from parallelrank import find_rank
from scipy.sparse import load_npz, rand
import tensorflow as tf
from time import perf_counter
from tqdm import tqdm
import argparse
import numpy as np


def main():
    # Parse arguements
    parser = argparse.ArgumentParser(
                prog='ParallelRankPerforamnceMeasurement',
                description='This script measures performance on data generated from ParallelRankDataGenerator')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-c', '--cpu', action='store_true')
    parser.add_argument('-m', '--memory', action='store_true')
    args = parser.parse_args()

    # Data generation parametres
    sides = get_sides(args.test)
    make_cpu = args.cpu
    memory_mode = args.memory

    read_dir = DATA_DIR
    main_dir = "parallelrank_perfs"
    create_dir(main_dir)
    
    # Warp up GPU
    print("Warming up GPU on 500x500 matrix")
    matrix = rand(500, 500, density=1e-3, format='csr', dtype=np.int8)
    matrix.data[:] = 1
    for i in tqdm(range(4)):
        find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
    
    # Measure performance on generated data
    print("Finding rank of matrices with different sides in directory \"{}\":".format(read_dir))
    print("Saving results in \"{}\":".format(main_dir))
    for side in sides:
        # Generate matrix with changing and constant width
        print("Finding rank for matrices with side", side)
        for nnz, shape in tqdm([
                (CONST_WIDTH * CONST_WIDTH, (CONST_WIDTH, side)), 
                (0.1 * side, (side, side)), 
                (0.01 * side, (side, side)), 
                (CONST_WIDTH * CONST_WIDTH, (side, side))
        ]):
            width, height = shape
            
            total_time = 0
            successful_repititions = 0
            
            # Use several repititions to get more precise results from average data
            for i in range(REPETITIONS):
                matrix = load_npz("{}/shape_{}x{}/matrix_nnz_{}_iter_{}.npz".format(read_dir, width, height, nnz, i))
                
                time_start = perf_counter()
                rank, mem = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0], log_memory=True)
                time_stop = perf_counter()
                time_spent = time_stop - time_start
                
                if memory_mode:
                    total_time += int(mem)
                else:
                    total_time += time_spent
                successful_repititions += 1
                
            np.save(
                "{}/algorithm_time_{}x{}_nnz_{}.npy".format(main_dir, width, height, nnz), 
                total_time / successful_repititions
            )
            
            if make_cpu:
                total_time = 0
                successful_repititions = 0

                for i in range(REPETITIONS):
                    matrix = load_npz("{}/shape_{}x{}/matrix_nnz_{}_iter_{}.npz".format(read_dir, width, height, nnz, i))
                    
                    time_start = perf_counter()
                    rank, mem = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0], algorithm='cpu', log_memory=True)
                    time_stop = perf_counter()
                    time_spent = time_stop - time_start
                    
                    if memory_mode:
                        total_time += int(mem)
                    else:
                        total_time += time_spent
                    successful_repititions += 1
                    
                np.save(
                    "{}/algorithm_time_{}x{}_nnz_{}_cpu.npy".format(main_dir, width, height, nnz), 
                    total_time / successful_repititions
                )
                
            
            if width * height > 10_000_000_000:
                continue
            
            total_time = 0
            successful_repititions = 0
            
            # Use several repititions to get more precise results from average data
            for i in range(REPETITIONS):
                matrix = load_npz("{}/shape_{}x{}/matrix_nnz_{}_iter_{}.npz".format(read_dir, width, height, nnz, i))
                
                time_start = perf_counter()
                rank, mem = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0], algorithm='gauss', log_memory=True)
                time_stop = perf_counter()
                time_spent = time_stop - time_start
                
                if memory_mode:
                    total_time += int(mem)
                else:
                    total_time += time_spent
                successful_repititions += 1
                
            np.save(
                "{}/algorithm_time_{}x{}_nnz_{}_gauss.npy".format(main_dir, width, height, nnz), 
                total_time / successful_repititions
            )
            
            if width * height > 1_000_000_000:
                continue
            
            total_time = 0
            successful_repititions = 0
            
            # Use several repititions to get more precise results from average data
            for i in range(REPETITIONS):
                matrix = load_npz("{}/shape_{}x{}/matrix_nnz_{}_iter_{}.npz".format(read_dir, width, height, nnz, i))
                
                time_start = perf_counter()
                tf.linalg.matrix_rank(matrix.todense(), tol=1e-5)
                time_stop = perf_counter()
                time_spent = time_stop - time_start
                
                total_time += time_spent
                successful_repititions += 1
                
            np.save(
                "{}/algorithm_time_{}x{}_nnz_{}_svd.npy".format(main_dir, width, height, nnz), 
                total_time / successful_repititions
            )

if __name__ == "__main__":
    main()
