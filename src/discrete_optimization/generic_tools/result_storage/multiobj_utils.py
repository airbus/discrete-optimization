#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np


class TupleFitness:
    vector_fitness: np.ndarray
    size: int

    def __init__(self, vector_fitness: np.ndarray, size: int):
        self.vector_fitness = vector_fitness
        self.size = size

    def distance(self, other: "TupleFitness") -> np.floating:
        return np.linalg.norm(self.vector_fitness - other.vector_fitness, ord=2)

    # if none of the two solution dominates the other one.
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TupleFitness)
            and not (self < other)
            and not (self > other)
        )

    def __le__(self, other: "TupleFitness") -> bool:
        return self < other or self == other

    def __ge__(self, other: "TupleFitness") -> bool:
        return self > other or self == other

    def __lt__(self, other: "TupleFitness") -> bool:
        return bool(
            (self.vector_fitness <= other.vector_fitness).all()
            and (self.vector_fitness < other.vector_fitness).any()
        )

    def __gt__(self, other: "TupleFitness") -> bool:
        return bool(
            (self.vector_fitness >= other.vector_fitness).all()
            and (self.vector_fitness > other.vector_fitness).any()
        )

    def __str__(self) -> str:
        return str(self.vector_fitness)

    def __mul__(self, other: float) -> "TupleFitness":
        return TupleFitness(other * self.vector_fitness, self.size)
