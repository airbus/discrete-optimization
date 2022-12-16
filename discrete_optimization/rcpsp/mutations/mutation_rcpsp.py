#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove
from discrete_optimization.generic_tools.do_problem import TypeAttribute
from discrete_optimization.generic_tools.mutations.mutation_util import (
    get_attribute_for_type,
)
from discrete_optimization.generic_tools.mutations.permutation_mutations import (
    Mutation,
    PermutationShuffleMutation,
    Problem,
    ShuffleMove,
    Solution,
    SwapsLocalMove,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution


class PermutationMutationRCPSP(Mutation):
    @staticmethod
    def build(
        problem: Problem, solution: Solution, **kwargs
    ) -> "PermutationMutationRCPSP":
        other_mutation = kwargs.get("other_mutation", PermutationShuffleMutation)
        other_mutation = other_mutation.build(problem, solution, **kwargs)
        return PermutationMutationRCPSP(
            problem, solution, other_mutation=other_mutation
        )

    def __init__(self, problem: Problem, solution: Solution, other_mutation: Mutation):
        self.problem = problem
        self.solution = solution
        self.other_mutation = other_mutation

    def mutate(self, solution: RCPSPSolution) -> Tuple[Solution, LocalMove]:
        s, lm = self.other_mutation.mutate(solution)
        try:
            s.standardised_permutation = s.generate_permutation_from_schedule()
            s._schedule_to_recompute = True
        except:
            s._schedule_to_recompute = True
        return s, lm

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, lm, fit = self.other_mutation.mutate_and_compute_obj(solution)
        try:
            s._schedule_to_recompute = True
            s.standardised_permutation = s.generate_permutation_from_schedule()
        except:
            s._schedule_to_recompute = True
        return s, lm, fit


class DeadlineMutationRCPSP(Mutation):
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
        self.full_predecessors = self.problem.graph.ancestors_map()

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        if "special_constraints" in self.problem.__dict__.keys():
            ls = [
                (
                    t,
                    solution.get_end_time(t)
                    - self.problem.special_constraints.end_times_window[t][1],
                )
                for t in self.problem.special_constraints.end_times_window
                if self.problem.special_constraints.end_times_window[t][1] is not None
                and solution.get_end_time(t)
                > self.problem.special_constraints.end_times_window[t][1]
            ]
            if len(ls) > 0:
                x = random.choice(ls)
                t = x[0]
                pred = [tt for tt in self.full_predecessors[t]] + [t]
                previous = list(getattr(solution, self.attribute))
                new = [
                    self.problem.index_task_non_dummy[tt]
                    for tt in pred
                    if tt in self.problem.index_task_non_dummy
                ]
                for x in previous:
                    if x not in new:
                        new += [x]
                sol = solution.lazy_copy()
                setattr(sol, self.attribute, new)
                return (
                    sol,
                    ShuffleMove(
                        self.attribute,
                        new_permutation=new,
                        prev_permutation=previous,
                    ),
                )
        swaps = np.random.randint(low=0, high=self.length - 1, size=(1, 2))
        move = SwapsLocalMove(
            self.attribute, [(swaps[i, 0], swaps[i, 1]) for i in range(1)]
        )
        next_sol = move.apply_local_move(solution)
        return next_sol, move

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        sol, move = self.mutate(solution)
        obj = self.problem.evaluate(sol)
        return sol, move, obj

    @staticmethod
    def build(
        problem: Problem, solution: Solution, **kwargs
    ) -> "DeadlineMutationRCPSP":
        return DeadlineMutationRCPSP(problem, solution)
