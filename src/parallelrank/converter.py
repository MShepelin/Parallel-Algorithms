from __future__ import print_function
import ctypes
import os

'''
Prints out the error message and quits the program.
msg -- Custom error message to show the user
'''
def print_help_exit(msg):
    error_msg = msg + '''
    How to use this:
    ParallelRank package:
    User Functions:
        find_rank(column_offsets, rows_indicies, rows, columns, max_attempts=None, algorithm='hom')
    For more information, please see README.md.
    '''
    
    #TODO: add arguements explanation

    raise Exception(error_msg)


# Searches the path and all its children for file named name
def find(name, path):#stackoverflow.com/questions/1724693/find-a-file-in-python
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)


def convert(prog, user_matrix): 
    num_rows, num_columns, num_entries, user_matrix = ravel_matrix(user_matrix)
    if user_matrix is None:
        print_help_exit("Matrix was not created")
        return

    user_matrix = (ctypes.c_float * num_entries)(*user_matrix)
    prog.findRank.restype = ctypes.c_int32
    res = prog.findRank(user_matrix, num_rows, num_columns)
    return res


def ravel_matrix(user_matrix):
    if len(user_matrix.shape)!=2:
        print_help_exit("Matrix must be 2-dimensional")
        return

    num_rows, num_columns = user_matrix.shape
    num_entries = num_rows * num_columns
    return num_rows, num_columns, num_entries, user_matrix.ravel()
