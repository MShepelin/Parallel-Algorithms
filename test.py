import parallelrank
import numpy as np

array = np.random.randint(0, 8, (1000, 1000))
print(parallelrank.findRank(array))


second_array = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [3, 4, 0, 0],
    [2, 2, 0, 0]
])
print(parallelrank.findRank(second_array))
