#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from BinPacking to RCPSP.

Clever mapping:
- Each item → task with duration=1
- Bin capacity → cumulative resource
- Item weight → resource requirement
- Incompatible items → virtual unary resources (capacity 1)

This allows using powerful RCPSP solvers for bin packing!
"""

from enum import Enum

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
    Objective,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)


class BinpackToGenericApproach(Enum):
    ALLOCATION = 0
    SCHEDULING = 1


class BinpackToGenericSchedulingTransformation(
    ProblemTransformation[
        BinPackProblem,
        BinPackSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform BinPacking to RCPSP.
    """

    def __init__(
        self, modeling: BinpackToGenericApproach = BinpackToGenericApproach.SCHEDULING
    ):
        self.modeling = modeling

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → RCPSP).

        This direction is EXACT: all constraints can be represented in RCPSP.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of bin packing as scheduling problem",
                "Incompatibility modeled via no-overlap set of constraints",
            ]
        )

    def transform_problem(
        self, source_problem: BinPackProblem
    ) -> GenericSchedulingImplProblem:
        """Transform BinPacking to RCPSP.

        Args:
            source_problem: BinPacking problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        if self.modeling == BinpackToGenericApproach.SCHEDULING:
            return GenericSchedulingImplProblem(
                horizon=source_problem.nb_items,
                durations_per_mode={i: {0: 1} for i in range(source_problem.nb_items)},
                resource_consumptions={
                    i: {0: {"capacity": int(source_problem.list_items[i].weight)}}
                    for i in range(source_problem.nb_items)
                },
                non_skill_cumulative_resources={
                    "capacity": [
                        (
                            i,
                            i + 1,
                            source_problem.list_bin_instances[i].bin_type.capacity,
                        )
                        for i in range(len(source_problem.list_bin_instances))
                    ]
                },
                no_overlap_sets={
                    frozenset([item1, item2])
                    for item1, item2 in source_problem.incompatible_items
                },
                forbidden_intervals={
                    item: [
                        (i, i + 1)
                        for i in range(len(source_problem.list_bin_instances))
                        if item
                        not in source_problem.list_bin_instances[i].compatible_items
                    ]
                    for item in range(source_problem.nb_items)
                },
                objective=Objective.MAKESPAN,
            )
        raise NotImplementedError()

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: BinPackProblem
    ) -> BinPackSolution:
        """Transform GenericSchedulingImpl solution back to BinPacking solution.

        Returns:
            Equivalent BinPacking solution

        """
        # Extract bin assignment from start times
        # Tasks scheduled at time t are assigned to bin t
        allocation = [0] * source_problem.nb_items

        for i, item in enumerate(source_problem.list_items):
            task_name = f"item_{item.index}"
            allocation[i] = solution.get_start_time(i)
        return BinPackSolution(problem=source_problem, allocation=allocation)
