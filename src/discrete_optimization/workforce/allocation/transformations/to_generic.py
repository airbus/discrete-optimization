#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Coloring to RCPSP.
"""

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.workforce.allocation.problem import (
    TeamAllocationProblem,
    TeamAllocationSolution,
)


class WorkforceAllocationToGenericSchedulingTransformation(
    ProblemTransformation[
        TeamAllocationProblem,
        TeamAllocationSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform AllocationProblem to RCPSP.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → RCPSP).

        This direction is EXACT: all constraints can be represented in RCPSP.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of workforce alloc problem as scheduling problem",
            ]
        )

    def transform_problem(
        self, source_problem: TeamAllocationProblem
    ) -> GenericSchedulingImplProblem:
        """Transform BinPacking to RCPSP.

        Args:
            source_problem: BinPacking problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        tasks = source_problem.tasks_list
        unary = source_problem.unary_resources_list
        time_window = {}
        for t in source_problem.schedule:
            st, end = source_problem.schedule[t]
            print(st, end)
            time_window[t] = (st, end, st, end)
        compatible_teams = source_problem.compute_compatibility_for_all_tasks()
        return GenericSchedulingImplProblem(
            horizon=max([x[1] for x in time_window.values()]),
            skills={"team"},
            unary_resources=unary,
            resource_consumptions={t: {0: {"team": 1}} for t in tasks},
            unary_resources_skills={ur: {"team": 1} for ur in unary},
            durations_per_mode={
                n: {0: source_problem.schedule[n][1] - source_problem.schedule[n][0]}
                for n in source_problem.tasks_list
            },
            time_windows=time_window,
            unary_resources_task_compatibility=compatible_teams,
            list_objective_computer=[
                UnaryResourcesUsedComputer(
                    weight_objective=1,
                    weight_per_unary_resource={ur: 1 for ur in unary},
                )
            ],
        )

    def back_transform_solution(
        self,
        solution: GenericSchedulingImplSolution,
        source_problem: TeamAllocationProblem,
    ) -> TeamAllocationSolution:
        """Transform GenericSchedulingImpl solution back to TeamAllocation solution.

        Returns:
            Equivalent TeamAllocation solution

        """
        # Extract bin assignment from start times
        # Tasks scheduled at time t are assigned to bin t
        teams = [-1] * len(source_problem.tasks_list)
        for i, item in enumerate(source_problem.tasks_list):
            if solution.is_present(item):
                allocated = solution.get_task_allocation(item)
                if len(allocated) >= 1:
                    team = list(allocated)[0]
                    teams[i] = source_problem.index_teams_name[team]
        return TeamAllocationSolution(problem=source_problem, allocation=teams)
