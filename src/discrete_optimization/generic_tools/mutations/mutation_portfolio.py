#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections.abc import Container
from typing import Union

import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove, Mutation
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)

logger = logging.getLogger(__name__)


class PortfolioMutation(Mutation):
    """Mutations portfolio.

    Randomly choose between available mutations.

    """

    def __init__(
        self,
        problem: Problem,
        list_mutations: list[Mutation],
        weight_mutations: Union[list[float], np.ndarray],
        **kwargs,
    ):
        super().__init__(problem=problem, **kwargs)
        self.list_mutations = list_mutations
        self.weight_mutation = weight_mutations
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation / np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutations)), dtype=np.int_)

    def choose_a_mutation(self) -> int:
        return int(np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0])

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        choice = self.choose_a_mutation()
        logger.debug(f"mutate with {self.list_mutations[choice]}")
        return self.list_mutations[choice].mutate(solution)

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> tuple[Solution, LocalMove, dict[str, float]]:
        choice = self.choose_a_mutation()
        return self.list_mutations[choice].mutate_and_compute_obj(solution)

    def __repr__(self):
        return f"{type(self).__name__}(list_mutations='{self.list_mutations}')"


def create_mutations_portfolio_from_problem(
    problem: Problem,
    selected_mutations: Container[type[Mutation]] | None = None,
    selected_attributes: Container[str] | None = None,
) -> PortfolioMutation:
    """Create a mutation mixing all mutations available in catalog for the solution attributes."""
    # available mutations for the encoding attributes specified by the problem/solution
    list_mutations = get_available_mutations(problem)

    def _filter(
        mutation_type: type[Mutation],
        attribute_name: str,
        selected_mutations: Container[type[Mutation]] | None = None,
        selected_attributes: Container[str] | None = None,
    ) -> bool:
        return (
            selected_attributes is None or attribute_name in selected_attributes
        ) and (selected_mutations is None or mutation_type in selected_mutations)

    list_built_mutations = [
        mutation_cls.build(problem=problem, attribute=attribute_name, **mutation_kwargs)
        for mutation_cls, mutation_kwargs, attribute_name in list_mutations
        if _filter(
            mutation_cls, attribute_name, selected_mutations, selected_attributes
        )
    ]

    # create a mixed mutation that sample one of the given mutations
    return PortfolioMutation(
        problem=problem,
        list_mutations=list_built_mutations,
        weight_mutations=np.ones((len(list_built_mutations))),
    )
