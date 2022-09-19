#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)


def test_tuplefitness():
    fitness_1 = TupleFitness(np.array([1, 2]), 2)
    fitness_2 = TupleFitness(np.array([2, 3]), 2)
    fitness_3 = TupleFitness(np.array([0, 3]), 2)
    assert fitness_1 <= fitness_2
    assert fitness_1 < fitness_2
    assert not (fitness_1 == fitness_2)
    assert not (fitness_1 > fitness_2)
    assert not (fitness_1 >= fitness_2)
    assert fitness_1 == fitness_3
    assert fitness_3 < fitness_2
    assert fitness_3 <= fitness_2
    assert not (fitness_3 > fitness_2)
    assert not (fitness_3 >= fitness_2)
    assert not (fitness_3 == fitness_2)


if __name__ == "__main__":
    test_tuplefitness()
