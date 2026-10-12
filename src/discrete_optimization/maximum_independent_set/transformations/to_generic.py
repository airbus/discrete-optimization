#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Knapsack to GenSchedulingImpl
"""

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)


class MisToGenericSchedulingTransformation(
    ProblemTransformation[
        MisProblem,
        MisSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform MIS to GenericSchedulingImpl.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (MIS → GenSchedulingImpl).

        This direction is EXACT: all constraints can be represented in GenSchedulingImpl.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of MIS as scheduling problem",
            ]
        )

    def transform_problem(
        self, source_problem: MisProblem
    ) -> GenericSchedulingImplProblem:
        """Transform MisProblem to GenericSchedulingImpl problem.

        Args:
            source_problem: MisProblem problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        tasks = [i for i in range(source_problem.number_nodes)]
        linear_constraints = [
            (
                [
                    (source_problem.nodes_to_index[e[0]], 1),
                    (source_problem.nodes_to_index[e[1]], 1),
                ],
                0,
                1,
            )
            for e in source_problem.edges
        ]
        from discrete_optimization.generic_tasks_tools.objectives.scheduled_tasks import (
            ScheduledTasksComputer,
        )

        return GenericSchedulingImplProblem(
            horizon=1,
            durations_per_mode={i: {0: 0} for i in tasks},
            optional_tasks=set(tasks),
            linear_cstrs_presence_tasks=linear_constraints,
            list_objective_computer=[
                ScheduledTasksComputer(
                    weight_objective=-1,  # Maximize.
                    weight_per_task={i: 1 for i in tasks},
                )
            ],
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: MisProblem
    ) -> MisSolution:
        """Transform GenericSchedulingImpl solution back to Knapsack solution.

        Returns:
            Equivalent Knapsack solution

        """
        chosen = [False for i in range(source_problem.number_nodes)]
        for i in solution.problem.tasks_list:
            if solution.is_present(i):
                chosen[i] = True
        return MisSolution(problem=source_problem, chosen=chosen)
