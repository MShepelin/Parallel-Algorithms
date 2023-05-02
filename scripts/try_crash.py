from parallelrank import find_rank
from scipy.sparse import load_npz
from time import perf_counter
import argparse

def main():
    matrix_path = "data/side_50000/matrix_1.npz"
     
    parser = argparse.ArgumentParser(
                prog='ParallelRankCrashTester',
                description='This program tests how program acts on a huge matrix (assumes you have file \"{}\")'.format(matrix_path))
    parser.parse_args()
   
    matrix = load_npz(matrix_path)
    print("Matrix has", len(matrix.indices), "indices and ", len(matrix.indptr), "offset values", flush=True)
    
    time_start = perf_counter()
    find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
    time_stop = perf_counter()

    print("Rank computation time in seconds (HUGE matrix):", time_stop-time_start)
