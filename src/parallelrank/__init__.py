import ctypes
from .converter import convert, printHelpAndExit
import numpy as np
import platform
import pathlib


# rename using _ instead of camel case to mimic tensorflow
def findRank(matrix): 
    if not isinstance(matrix, np.ndarray):
        printHelpAndExit("Error: matrix should be np.ndarray")
    
    prog = None

    path= str(pathlib.Path(__file__).parents[0].with_name('libparallel_rank.so'))
    if platform.system() == 'Windows':
        path = str(pathlib.Path(__file__).parents[0].with_name('parallel_rank.dll'))
    if platform.system() == 'Darwin':
        path = str(pathlib.Path(__file__).parents[0].with_name('parallel_rank.dylib'))

    if path != None:
        prog = ctypes.cdll.LoadLibrary(path)
    else:
        if platform.system() == "Windows":
            raise Exception("Could not locate parallel_rank.dll file, please check README.md for details.")
        elif platform.system() == "Darwin":
            raise Exception("Could not locate parallel_rank.dylib file, please check README.md for details.")
        else:
            raise Exception("Could not locate parallel_rank.so file, please check README.md for details.")

    rank = convert(prog, matrix)
    return rank
