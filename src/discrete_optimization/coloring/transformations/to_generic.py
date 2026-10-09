#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Coloring to RCPSP.
"""

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.makespan import (
    MakespanObjectiveComputer,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)


class ColoringToGenericSchedulingTransformation(
    ProblemTransformation[
        ColoringProblem,
        ColoringSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform Coloring to RCPSP.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → RCPSP).

        This direction is EXACT: all constraints can be represented in RCPSP.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of coloring as scheduling problem",
                "Incompatibility modeled via no-overlap set of constraints",
            ]
        )

    def transform_problem(
        self, source_problem: ColoringProblem
    ) -> GenericSchedulingImplProblem:
        """Transform BinPacking to RCPSP.

        Args:
            source_problem: BinPacking problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        nb_max_colors = len(source_problem.tasks_list)
        time_window = {}
        if source_problem.has_constraints_coloring:
            for t in source_problem.constraints_coloring.nodes_fixed():
                val = source_problem.constraints_coloring.color_constraint[t]
                time_window[t] = (val, val, val + 1, val + 1)
        return GenericSchedulingImplProblem(
            horizon=nb_max_colors,
            durations_per_mode={n: {0: 1} for n in source_problem.tasks_list},
            time_windows=time_window,
            no_overlap_sets={
                frozenset([e[0], e[1]]) for e in source_problem.graph.edges
            },
            list_objective_computer=[MakespanObjectiveComputer(weight_objective=1)],
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: ColoringProblem
    ) -> ColoringSolution:
        """Transform GenericSchedulingImpl solution back to BinPacking solution.

        Returns:
            Equivalent BinPacking solution

        """
        # Extract bin assignment from start times
        # Tasks scheduled at time t are assigned to bin t
        colors = [-1] * len(source_problem.tasks_list)
        for i, item in enumerate(source_problem.tasks_list):
            if solution.is_present(i):
                colors[i] = solution.get_start_time(i)
        return ColoringSolution(problem=source_problem, colors=colors)
