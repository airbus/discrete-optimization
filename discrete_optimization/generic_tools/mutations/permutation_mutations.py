#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove, Mutation
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    TypeAttribute,
)
from discrete_optimization.generic_tools.mutations.mutation_util import (
    get_attribute_for_type,
)


class ShuffleMove(LocalMove):
    def __init__(
        self,
        attribute: str,
        new_permutation: Union[List[int], np.ndarray],
        prev_permutation: Union[List[int], np.ndarray],
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


class PermutationShuffleMutation(Mutation):
    @staticmethod
    def build(
        problem: Problem, solution: Solution, **kwargs: Any
    ) -> "PermutationShuffleMutation":
        return PermutationShuffleMutation(problem, solution)

    def __init__(
        self, problem: Problem, solution: Solution, attribute: Optional[str] = None
    ):
        self.problem = problem
        register = solution.get_attribute_register(problem)
        if attribute is None:
            self.attribute = get_attribute_for_type(register, TypeAttribute.PERMUTATION)
        else:
            self.attribute = attribute
        self.range_shuffle = register.dict_attribute_to_type[self.attribute]["range"]
        self.range_int = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        previous = list(getattr(solution, self.attribute))
        random.shuffle(self.range_int)
        new = [previous[i] for i in self.range_int]
        sol = solution.lazy_copy()
        setattr(sol, self.attribute, new)
        return (
            sol,
            ShuffleMove(self.attribute, new_permutation=new, prev_permutation=previous),
        )

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class PermutationPartialShuffleMutation(Mutation):
    @staticmethod
    def build(
        problem: Problem, solution: Solution, **kwargs: Any
    ) -> "PermutationPartialShuffleMutation":
        return PermutationPartialShuffleMutation(
            problem,
            solution,
            attribute=kwargs.get("attribute", None),
            proportion=kwargs.get("proportion", 0.3),
        )

    def __init__(
        self,
        problem: Problem,
        solution: Solution,
        attribute: Optional[str] = None,
        proportion: float = 0.3,
    ):
        self.problem = problem
        register = solution.get_attribute_register(problem)
        if attribute is None:
            self.attribute = get_attribute_for_type(register, TypeAttribute.PERMUTATION)
        else:
            self.attribute = attribute
        self.range_shuffle = register.dict_attribute_to_type[self.attribute]["range"]
        self.n_to_move = int(proportion * len(self.range_shuffle))
        self.range_int = list(range(self.n_to_move))
        self.range_int_total = list(range(len(self.range_shuffle)))

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
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

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class SwapsLocalMove(LocalMove):
    def __init__(self, attribute: str, list_index_swap: List[Tuple[int, int]]):
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


class PermutationSwap(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs: Any) -> "PermutationSwap":
        return PermutationSwap(
            problem,
            solution,
            attribute=kwargs.get("attribute", None),
            nb_swap=kwargs.get("nb_swap", 1),
        )

    def __init__(
        self,
        problem: Problem,
        solution: Solution,
        attribute: Optional[str] = None,
        nb_swap: int = 1,
    ):
        self.problem = problem
        self.nb_swap = nb_swap
        register = solution.get_attribute_register(problem)
        if attribute is None:
            self.attribute = get_attribute_for_type(register, TypeAttribute.PERMUTATION)
        else:
            self.attribute = attribute
        self.length = len(register.dict_attribute_to_type[self.attribute]["range"])

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        swaps = np.random.randint(low=0, high=self.length - 1, size=(self.nb_swap, 2))
        move = SwapsLocalMove(
            self.attribute, [(swaps[i, 0], swaps[i, 1]) for i in range(self.nb_swap)]
        )
        next_sol = move.apply_local_move(solution)
        return next_sol, move

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj


class TwoOptMove(LocalMove):
    def __init__(self, attribute: str, index_2opt: List[Tuple[int, int]]):
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


class TwoOptMutation(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs: Any) -> "TwoOptMutation":
        return TwoOptMutation(
            problem, solution, attribute=kwargs.get("attribute", None)
        )

    def __init__(
        self, problem: Problem, solution: Solution, attribute: Optional[str] = None
    ):
        self.problem = problem
        register = solution.get_attribute_register(problem)
        if attribute is None:
            self.attribute = get_attribute_for_type(register, TypeAttribute.PERMUTATION)
        else:
            self.attribute = attribute
        self.length = len(register.dict_attribute_to_type[self.attribute]["range"])

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        i = random.randint(0, self.length - 2)
        j = random.randint(i + 1, self.length - 1)
        two_opt_move = TwoOptMove(self.attribute, [(i, j)])
        new_sol = two_opt_move.apply_local_move(solution)
        return new_sol, two_opt_move

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj
