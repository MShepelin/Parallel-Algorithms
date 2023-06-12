import ctypes
from .converter import print_help_exit
from .globals import PROG
import numpy as np
import platform
import pathlib

__all__ = ['find_rank']

def load_DLL():
    global PROG
    
    if PROG is not None:
        return
    
    path = str(pathlib.Path(__file__).parents[0].with_name('libparallel_rank.so'))
    if platform.system() == 'Windows':
        path = str(pathlib.Path(__file__).parents[0].with_name('parallel_rank.dll'))
    if platform.system() == 'Darwin':
        path = str(pathlib.Path(__file__).parents[0].with_name('parallel_rank.dylib'))

    if path != None:
        PROG = ctypes.cdll.LoadLibrary(path)
    else:
        if platform.system() == "Windows":
            raise Exception("Could not locate parallel_rank.dll file, please check README.md for details.")
        elif platform.system() == "Darwin":
            raise Exception("Could not locate parallel_rank.dylib file, please check README.md for details.")
        else:
            raise Exception("Could not locate parallel_rank.so file, please check README.md for details.")


def find_rank(column_offsets, rows_indicies, rows, columns, max_attempts=None, algorithm='mix', log_memory=False):
    if max_attempts is None:
        max_attempts = min(rows, columns)
    
    # Recommended max_attempts = columns
    global PROG
    
    assert type(rows) == int and type(columns) == int
    
    if not isinstance(column_offsets, np.ndarray) or not isinstance(rows_indicies, np.ndarray):
        print_help_exit("Error: matrix should be np.ndarray")
        
    if column_offsets.dtype != np.int32 or rows_indicies.dtype != np.int32:
        print_help_exit("Error: matrix should be have dtype np.int32")
        
    if len(column_offsets.shape) != 1 or len(rows_indicies.shape) != 1:
        print_help_exit("Error: column_offsets and rows_indicies should be 1D arrays")
        
    if algorithm != 'hom' and algorithm != 'gauss' and algorithm != 'cpu' and algorithm != 'mix':
        print_help_exit("Error: valid algorithm options are: \"hom\",\"gauss\",\"cpu\",\"mix\"")
    
    load_DLL()

    c_column_offsets = (ctypes.c_int32 * len(column_offsets))(*column_offsets)
    c_rows_indicies = (ctypes.c_int32 * len(rows_indicies))(*rows_indicies)
    
    memory_consumption_raw = ctypes.c_int32(0)
    memory_consumption = ctypes.byref(memory_consumption_raw)
    rank = 0
    
    if algorithm == 'mix':
        if rows * columns >= 1_000_000:
            algorithm = 'hom'
        else:
            algorithm = 'cpu'
    
    if algorithm == 'hom':
        PROG.find_rank.restype = ctypes.c_int32
        rank = PROG.find_rank(
            c_column_offsets, 
            len(column_offsets), 
            c_rows_indicies, 
            len(rows_indicies), 
            columns, 
            rows, 
            max_attempts,
            memory_consumption
        )
    elif algorithm == 'gauss':
        PROG.find_rank_gauss.restype = ctypes.c_int32
        rank = PROG.find_rank_gauss(
            c_column_offsets, 
            len(column_offsets), 
            c_rows_indicies, 
            len(rows_indicies), 
            columns, 
            rows,
            memory_consumption
        )
    elif algorithm == 'cpu':
        PROG.find_rank_cpu.restype = ctypes.c_int32
        rank = PROG.find_rank_cpu(
            c_column_offsets, 
            len(column_offsets), 
            c_rows_indicies, 
            len(rows_indicies), 
            columns, 
            rows, 
            max_attempts,
            memory_consumption
        )
        
    if log_memory:
        return rank, memory_consumption_raw.value
    else:
        return rank
