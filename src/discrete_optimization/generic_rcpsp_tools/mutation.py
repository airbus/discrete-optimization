#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_rcpsp_tools.attribute_type import PermutationRcpsp
from discrete_optimization.generic_rcpsp_tools.typing import (
    ANY_RCPSP_FOR_MUTATION,
    ANY_SOLUTION_FOR_MUTATION,
    is_instance_any_rcpsp_solution,
)
from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.mutations.mutation_permutation import (
    ShuffleMove,
    ShuffleMutation,
    SwapsLocalMove,
)


class RcpspMutation(SingleAttributeMutation):
    """Wrapper around other mutations which makes recompute the schedule."""

    attribute_type_cls = PermutationRcpsp
    attribute_type: PermutationRcpsp
    problem: ANY_RCPSP_FOR_MUTATION

    @classmethod
    def build(
        cls,
        problem: Problem,
        other_mutation_cls: type[SingleAttributeMutation],
        **kwargs: Any,
    ) -> RcpspMutation:
        built_other_mutation = other_mutation_cls.build(problem, **kwargs)
        return cls(problem, other_mutation=built_other_mutation)

    def __init__(
        self,
        problem: ANY_RCPSP_FOR_MUTATION,
        other_mutation: SingleAttributeMutation,
        **kwargs: Any,
    ):
        attribute = other_mutation.attribute
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.other_mutation = other_mutation

    def mutate(
        self, solution: ANY_SOLUTION_FOR_MUTATION
    ) -> tuple[ANY_SOLUTION_FOR_MUTATION, LocalMove]:  # type: ignore
        s: ANY_SOLUTION_FOR_MUTATION
        s, lm = self.other_mutation.mutate(solution)  # type: ignore
        _recompute_schedule_n_standardised_permutation(s)
        return s, lm

    def mutate_and_compute_obj(  # type: ignore
        self, solution: ANY_SOLUTION_FOR_MUTATION
    ) -> tuple[ANY_SOLUTION_FOR_MUTATION, LocalMove, dict[str, float]]:
        s: ANY_SOLUTION_FOR_MUTATION
        s, lm, fit = self.other_mutation.mutate_and_compute_obj(solution)  # type: ignore
        _recompute_schedule_n_standardised_permutation(s)
        return s, lm, fit

    def __repr__(self):
        return f"{type(self).__name__}(other_mutation='{self.other_mutation}')"


def _recompute_schedule_n_standardised_permutation(solution: ANY_SOLUTION_FOR_MUTATION):
    try:
        solution._schedule_to_recompute = True
        solution.standardised_permutation = (
            solution.generate_permutation_from_schedule()
        )  # type: ignore
    except:
        solution._schedule_to_recompute = True


class DeadlineRcpspMutation(SingleAttributeMutation):
    attribute_type_cls = PermutationRcpsp
    attribute_type: PermutationRcpsp
    problem: ANY_RCPSP_FOR_MUTATION

    def __init__(
        self,
        problem: ANY_RCPSP_FOR_MUTATION,
        attribute: Optional[str] = None,
        nb_swap: int = 1,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.nb_swap = nb_swap
        self.length = self.attribute_type.length
        try:
            self.full_predecessors = self.problem.graph.ancestors_map()  # type: ignore
        except:
            pass

    def mutate(
        self, solution: ANY_SOLUTION_FOR_MUTATION
    ) -> tuple[ANY_SOLUTION_FOR_MUTATION, LocalMove]:  # type: ignore
        if not is_instance_any_rcpsp_solution(solution):
            raise ValueError("solution must be an rcsp solution (of any kind)")
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
            self.attribute, [(int(swaps[i, 0]), int(swaps[i, 1])) for i in range(1)]
        )
        next_sol = move.apply_local_move(solution)
        return next_sol, move
