#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove, Mutation
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)


class BasicPortfolioMutation(Mutation):
    def __init__(
        self,
        list_mutation: list[Mutation],
        weight_mutation: Union[list[float], np.ndarray],
    ):
        self.list_mutation = list_mutation
        self.weight_mutation = weight_mutation
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation / np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutation)), dtype=np.int_)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate(solution)

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> tuple[Solution, LocalMove, dict[str, float]]:
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate_and_compute_obj(solution)


def create_mixed_mutation_from_problem_and_solution(
    problem: Problem, solution: Solution
) -> BasicPortfolioMutation:
    """Create a mutation mixing all mutations available in catalog for the solution attributes."""
    # available mutations for the encoding attributes specified by the problem/solution
    _, list_mutation = get_available_mutations(problem)
    list_mutation = [
        mutate[0].build(problem, solution, **mutate[1]) for mutate in list_mutation
    ]
    # create a mixed mutation that sample one of the given mutations
    return BasicPortfolioMutation(list_mutation, np.ones((len(list_mutation))))
