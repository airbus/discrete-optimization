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
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution


class KnapsackToGenericSchedulingTransformation(
    ProblemTransformation[
        KnapsackProblem,
        KnapsackSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform Knapsack to GenericSchedulingImpl.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Knapsack → GenSchedulingImpl).

        This direction is EXACT: all constraints can be represented in GenSchedulingImpl.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of Knapsack as scheduling problem",
            ]
        )

    def transform_problem(
        self, source_problem: KnapsackProblem
    ) -> GenericSchedulingImplProblem:
        """Transform KnapsackProblem to GenericSchedulingImpl problem.

        Args:
            source_problem: KnapsackProblem problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        tasks = [i for i in range(source_problem.nb_items)]
        non_renewable_resource = {"weight": source_problem.max_capacity}
        resource_consumption = {
            i: {0: {"weight": source_problem.list_items[i].weight}} for i in tasks
        }
        from discrete_optimization.generic_tasks_tools.objectives.scheduled_tasks import (
            ScheduledTasksComputer,
        )

        return GenericSchedulingImplProblem(
            horizon=1,
            durations_per_mode={i: {0: 0} for i in tasks},
            resource_consumptions=resource_consumption,
            non_renewable_resources=non_renewable_resource,
            optional_tasks=set(tasks),
            list_objective_computer=[
                ScheduledTasksComputer(
                    weight_objective=-1,  # Maximize.
                    weight_per_task={
                        i: source_problem.list_items[i].value for i in tasks
                    },
                )
            ],
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: KnapsackProblem
    ) -> KnapsackSolution:
        """Transform GenericSchedulingImpl solution back to Knapsack solution.

        Returns:
            Equivalent Knapsack solution

        """
        list_taken = [False for i in range(source_problem.nb_items)]
        for i in solution.problem.tasks_list:
            if solution.is_present(i):
                list_taken[i] = True
        return KnapsackSolution(problem=source_problem, list_taken=list_taken)
