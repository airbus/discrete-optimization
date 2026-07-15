#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Lot sizing specific mutations for capacitated multi-item problems.

Implements GPI (Generalized Pairwise Interchange) moves from:
Ceschia, Di Gaspero, Schaerf (2017) - "Solving discrete lot-sizing and
scheduling by simulated annealing and mixed integer programming"
"""

import random
from typing import Any, Optional

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.encoding_register import ListInteger


class GPIInsertMutation(SingleAttributeMutation):
    """GPI Insert move: move element from position i to position j.

    Example:
        [A, B, C, D, E] with i=1, j=3
        -> [A, C, D, B, E]  (remove B, shift C and D left, insert B at position 3)
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.attribute_type: ListInteger

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))

        size = len(vector)
        if size <= 1:
            return s2, LocalMoveDefault(solution, s2)

        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        if i != j:
            element = vector.pop(i)
            vector.insert(j, element)

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPISwapMutation(SingleAttributeMutation):
    """GPI Swap move: swap elements at positions i and j.

    Example:
        [A, B, C, D, E] with i=1, j=3
        -> [A, D, C, B, E]  (swap B and D)
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.attribute_type: ListInteger

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = list(getattr(s2, self.attribute))

        size = len(vector)
        if size <= 1:
            return s2, LocalMoveDefault(solution, s2)

        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        if i != j:
            vector[i], vector[j] = vector[j], vector[i]

        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)


class GPIMixedMutation(SingleAttributeMutation):
    """Mixed GPI mutation combining Insert and Swap moves.

    Randomly chooses between Insert (with probability beta) and Swap
    (with probability 1-beta). Default beta=0.7 matches the paper.
    """

    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        beta: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.beta = beta
        self.insert_mutation = GPIInsertMutation(problem, attribute, **kwargs)
        self.swap_mutation = GPISwapMutation(problem, attribute, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        if random.random() < self.beta:
            return self.insert_mutation.mutate(solution)
        else:
            return self.swap_mutation.mutate(solution)
