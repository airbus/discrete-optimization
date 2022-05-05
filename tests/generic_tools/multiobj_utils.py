import time

import numpy as np
from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)


def main():
    fitness_1 = TupleFitness(np.array([1, 2]), 2)
    fitness_2 = TupleFitness(np.array([2, 3]), 2)
    fitness_3 = TupleFitness(np.array([0, 3]), 2)
    t = time.time()
    print(fitness_1 <= fitness_2, " should be True")
    print(fitness_1 < fitness_2, " should be True")
    print(fitness_1 == fitness_2, " should be False")
    print(fitness_1 > fitness_2, " should be False")
    print(fitness_1 >= fitness_2, " should be False")
    print(fitness_1 == fitness_3, " should be True")
    print(fitness_3 < fitness_2, " should be True")
    print(fitness_3 <= fitness_2, " should be True")
    print(fitness_3 > fitness_2, " should be False")
    print(fitness_3 >= fitness_2, " should be False")
    print(fitness_3 == fitness_2, " should be False")
    t_end = time.time()
    print(t_end - t, " secs")


if __name__ == "__main__":
    main()
