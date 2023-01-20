from __future__ import print_function
import numpy as np
import scipy.sparse as sps
import ctypes
import math
import sys
import re
import os

'''
Prints out the error message and quits the program.
msg -- Custom error message to show the user
'''
def printHelpAndExit(msg):
    error_msg = msg + '''
    How to use this:
    parallelrank package:
    User Functions:
        run(matrix or file_name, thread_num[optional])
                First Argument: Could be either of the following but not both
                    matrix: Must be a 2-dimensional numpy array
                    file_name: Must be of type string
                Second Argument: Optional parameter
                    thread_num: Number of threads for paralleling (positive integer)
                    
    For more information, please see README.md.
    '''

    raise Exception(error_msg)


# Searches the path and all its children for file named name
def find(name, path):#stackoverflow.com/questions/1724693/find-a-file-in-python
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)


def convert(prog, user_matrix): 
    num_rows, num_columns, num_entries, user_matrix = ravel_matrix(user_matrix)
    if user_matrix is None:
        printHelpAndExit("Matrix was not created")
        return

    user_matrix = (ctypes.c_float * num_entries)(*user_matrix)
    prog.findRank.restype = ctypes.c_int32
    res = prog.findRank(user_matrix, num_rows, num_columns)
    return res


def ravel_matrix(user_matrix):
    if len(user_matrix.shape)!=2:
        printHelpAndExit("Matrix must be 2-dimensional")
        return

    num_rows, num_columns = user_matrix.shape
    num_entries = num_rows * num_columns
    return num_rows, num_columns, num_entries, user_matrix.ravel()
