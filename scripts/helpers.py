import os


SIDES = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
#1_000_000, 10_000_000 [100, 1_000, 10_000, 100_000] #
REPETITIONS = 5
CONST_WIDTH = 100
DATA_DIR = "parallelrank_data"


def get_sides(is_test):
    return SIDES[:2] if is_test else SIDES


# Try to create directory to store matrices
def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        return
