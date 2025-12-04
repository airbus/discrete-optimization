#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.encoding_register import Permutation


class _BasePermutationMutation(SingleAttributeMutation):
    """Base class for permutation mutations."""

    attribute_type_cls = Permutation
    attribute_type: Permutation

    def __init__(
        self, problem: Problem, attribute: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.range_shuffle = self.attribute_type.range
        self.length = len(self.range_shuffle)


class ShuffleMove(LocalMove):
    def __init__(
        self,
        attribute: str,
        new_permutation: Union[list[int], np.ndarray],
        prev_permutation: Union[list[int], np.ndarray],
    ):
        self.attribute = attribute
        self.permutation = new_permutation
        self.prev_permutation = prev_permutation

    def apply_local_move(self, solution: Solution) -> Solution:
        setattr(solution, self.attribute, self.permutation)
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        setattr(solution, self.attribute, self.prev_permutation)
        return solution


class ShuffleMutation(_BasePermutationMutation):
    def __init__(
        self, problem: Problem, attribute: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.range_int = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        previous = list(getattr(solution, self.attribute))
        random.shuffle(self.range_int)
        new = [previous[i] for i in self.range_int]
        sol = solution.lazy_copy()
        setattr(sol, self.attribute, new)
        return (
            sol,
            ShuffleMove(self.attribute, new_permutation=new, prev_permutation=previous),
        )


class PartialShuffleMutation(_BasePermutationMutation):
    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        proportion: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.n_to_move = int(proportion * len(self.range_shuffle))
        self.range_int = list(range(self.n_to_move))
        self.range_int_total = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        previous = deepcopy(getattr(solution, self.attribute))
        random.shuffle(self.range_int_total)
        int_to_move = self.range_int_total[: self.n_to_move]
        random.shuffle(self.range_int)
        new = getattr(solution, self.attribute)
        for k in range(self.n_to_move):
            new[int_to_move[k]] = previous[int_to_move[self.range_int[k]]]
        sol = solution.lazy_copy()
        setattr(sol, self.attribute, new)
        return sol, ShuffleMove(self.attribute, new, previous)


class SwapsLocalMove(LocalMove):
    def __init__(self, attribute: str, list_index_swap: list[tuple[int, int]]):
        self.attribute = attribute
        self.list_index_swap = list_index_swap

    def apply_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i1, i2 in self.list_index_swap:
            v1, v2 = current[i1], current[i2]
            current[i1], current[i2] = v2, v1
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i1, i2 in self.list_index_swap:
            v1, v2 = current[i1], current[i2]
            current[i1], current[i2] = v2, v1
        return solution


class SwapMutation(_BasePermutationMutation):
    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        nb_swap: int = 1,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.nb_swap = nb_swap

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        swaps = np.random.randint(low=0, high=self.length - 1, size=(self.nb_swap, 2))
        move = SwapsLocalMove(
            self.attribute,
            [(int(swaps[i, 0]), int(swaps[i, 1])) for i in range(self.nb_swap)],
        )
        next_sol = move.apply_local_move(solution)
        return next_sol, move


class TwoOptMove(LocalMove):
    def __init__(self, attribute: str, index_2opt: list[tuple[int, int]]):
        self.attribute = attribute
        self.index_2opt = index_2opt

    def apply_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i, j in self.index_2opt:
            current = current[:i] + current[i : j + 1][::-1] + current[j + 1 :]
        setattr(solution, self.attribute, current)
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        current = getattr(solution, self.attribute)
        for i, j in self.index_2opt[::-1]:
            current = current[:i] + current[i : j + 1][::-1] + current[j + 1 :]
        setattr(solution, self.attribute, current)
        return solution


class TwoOptMutation(_BasePermutationMutation):
    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        i = random.randint(0, self.length - 2)
        j = random.randint(i + 1, self.length - 1)
        two_opt_move = TwoOptMove(self.attribute, [(i, j)])
        new_sol = two_opt_move.apply_local_move(solution)
        return new_sol, two_opt_move
