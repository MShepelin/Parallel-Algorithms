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
