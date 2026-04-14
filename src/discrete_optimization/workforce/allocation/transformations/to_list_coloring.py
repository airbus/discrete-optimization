#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Allocation to List Coloring (direct approach)."""

from typing import Optional

from discrete_optimization.coloring.list_coloring_problem import ListColoringProblem
from discrete_optimization.coloring.problem import ColoringSolution
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationMetadata,
    lossy_transformation,
)
from discrete_optimization.workforce.allocation.problem import (
    TeamAllocationProblem,
    TeamAllocationSolution,
)


class WorkforceAllocationToListColoringTransformation(
    ProblemTransformation[
        TeamAllocationProblem,
        TeamAllocationSolution,
        ListColoringProblem,
        ColoringSolution,
    ]
):
    """Transform Workforce Allocation to List Coloring (direct approach).

    Mapping:
    - Tasks → Nodes
    - Teams → Colors (0, 1, 2, ..., nb_teams-1)
    - all_diff_allocation → Edges between tasks
    - Available teams per task → allowed_colors[task]
    - Minimize number of teams → Minimize number of colors

    This is the cleaner approach using ListColoringProblem directly,
    without dummy nodes.

    This transformation is LOSSY:
    - Forward (problem): LOSSY - calendar and same_allocation constraints lost
    - Backward (solution): EXACT - color assignments = team assignments
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Allocation → ListColoring).

        This direction is LOSSY but cleaner than the dummy node approach.
        """
        losses = [
            InformationLoss(
                name="calendar_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Team availability calendars and temporal constraints",
                reason="List coloring has no concept of time or availability",
                impact=LossImpact.MAJOR,
                workaround="Pre-compute allowed_colors based on calendar compatibility",
            ),
            InformationLoss(
                name="same_allocation",
                loss_type=LossType.CONSTRAINT,
                description="Tasks that must be assigned to the same team",
                reason="List coloring cannot enforce groups with same color",
                impact=LossImpact.MAJOR,
                workaround="Pre-merge tasks in same_allocation groups into single nodes",
            ),
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "all_diff_allocation constraints modeled as edges",
                "allowed_allocation maps to allowed_colors per node",
                "No calendar constraints (or pre-filtered)",
                "No same_allocation constraints (or pre-processed)",
            ],
            use_cases=[
                "Allocation problems with explicit allowed teams per task",
                "Direct modeling without dummy nodes",
                "Cleaner representation for list coloring solvers",
            ],
            warnings=[
                "Calendar constraints not enforced (pre-filter allowed_colors)",
                "Same_allocation constraints will be violated",
                "Verify solution feasibility in original problem",
            ],
        )

    def transform_problem(
        self, source_problem: TeamAllocationProblem
    ) -> ListColoringProblem:
        """Transform Workforce Allocation to List Coloring.

        Args:
            source_problem: TeamAllocationProblem instance

        Returns:
            Equivalent ListColoringProblem

        """
        # Step 1: Build conflict graph from activity graph
        if source_problem.graph_activity is not None:
            graph = source_problem.graph_activity
        else:
            # Build graph from all_diff_allocation constraints
            nodes = [(task, {}) for task in source_problem.activities_name]
            edges = []

            if source_problem.allocation_additional_constraint is not None:
                all_diff = (
                    source_problem.allocation_additional_constraint.all_diff_allocation
                )
                if all_diff is not None:
                    for task_group in all_diff:
                        task_list = list(task_group)
                        for i in range(len(task_list)):
                            for j in range(i + 1, len(task_list)):
                                edges.append((task_list[i], task_list[j], {}))
                                edges.append((task_list[j], task_list[i], {}))

            graph = Graph(
                nodes=nodes,
                edges=edges,
                undirected=True,
                compute_predecessors=False,
            )

        # Step 2: Build allowed_colors from graph_allocation
        # In the bipartite graph, absence of edge means allowed allocation
        allowed_colors = {}

        for task in source_problem.activities_name:
            # Get all teams
            all_teams = set(range(source_problem.number_of_teams))

            # Find forbidden teams from graph_allocation
            forbidden_teams = set()

            if source_problem.graph_allocation is not None:
                if task in source_problem.graph_allocation.nodes_infos_dict:
                    # Get neighbors of this task in bipartite graph
                    # Edges in allocation graph represent FORBIDDEN allocations
                    if hasattr(source_problem.graph_allocation, "graph_nx"):
                        neighbors = list(
                            source_problem.graph_allocation.graph_nx.neighbors(task)
                        )
                        for neighbor in neighbors:
                            if neighbor in source_problem.teams_name:
                                team_idx = source_problem.teams_name.index(neighbor)
                                forbidden_teams.add(team_idx)

            # Allowed teams = all teams - forbidden teams
            allowed_teams = all_teams - forbidden_teams

            # Map to colors (color k = team k)
            allowed_colors[task] = allowed_teams

        return ListColoringProblem(
            graph=graph,
            allowed_colors=allowed_colors,
            subset_nodes=None,  # All nodes are tasks (no dummy nodes)
            constraints_coloring=None,
        )

    def back_transform_solution(
        self, solution: ColoringSolution, source_problem: TeamAllocationProblem
    ) -> TeamAllocationSolution:
        """Transform List Coloring solution back to Workforce Allocation solution.

        Args:
            solution: List Coloring solution
            source_problem: Original TeamAllocationProblem

        Returns:
            Equivalent TeamAllocationSolution

        """
        # Color directly maps to team index
        # Color k = team k
        allocation = list(solution.colors)

        return TeamAllocationSolution(
            problem=source_problem,
            allocation=allocation,
        )

    def forward_transform_solution(
        self, solution: TeamAllocationSolution, target_problem: ListColoringProblem
    ) -> Optional[ColoringSolution]:
        """Transform Allocation solution to List Coloring solution (for warmstart).

        Args:
            solution: TeamAllocationSolution
            target_problem: Target ListColoringProblem

        Returns:
            Equivalent ColoringSolution

        """
        # Team index directly maps to color
        colors = list(solution.allocation)

        return ColoringSolution(
            problem=target_problem,
            colors=colors,
            nb_color=None,  # Will be recomputed
            nb_violations=None,  # Will be recomputed
        )
