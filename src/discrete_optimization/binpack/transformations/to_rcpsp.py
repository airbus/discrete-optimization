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

from typing import Optional

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution


class BinpackToRcpspTransformation(
    ProblemTransformation[BinPackProblem, BinPackSolution, RcpspProblem, RcpspSolution]
):
    """Transform BinPacking to RCPSP.

    Mapping strategy:
    - Each item → task with duration=1
    - Bin capacity → cumulative resource "BinCapacity"
    - Item weight → resource requirement for that task
    - Time slot t → bin t (items at same time = same bin)
    - Incompatible items → virtual resources of capacity 1

    Key insight: Tasks scheduled at the same time (time slot t) go in bin t.
    Cumulative resource constraint ensures bin capacity is respected.
    Virtual resources ensure incompatible items can't be in same bin (same time).

    This transformation is EXACT in both directions:
    - Forward (problem): All constraints can be represented (incompatibility via virtual resources)
    - Backward (solution): Time slots map directly to bin assignments

    Example:
        # >>> # Bin packing: 3 items, capacity 10
        # >>> # Item 0: weight 5, Item 1: weight 3, Item 2: weight 6
        # >>> # Items 0 and 2 are incompatible
        # >>> problem = BinPackProblem(
        # ...     list_items=[ItemBinPack(0, 5), ItemBinPack(1, 3), ItemBinPack(2, 6)],
        # ...     capacity_bin=10,
        # ...     incompatible_items={(0, 2)}
        # ... )
        # >>>
        # >>> # RCPSP transformation:
        # >>> # - 3 tasks with duration=1
        # >>> # - Resource "BinCapacity" with capacity=10
        # >>> # - Task 0 needs 5, Task 1 needs 3, Task 2 needs 6
        # >>> # - Resource "Incompatible_0_2" with capacity=1 (both tasks consume it)
        # >>> # - Tasks at time 0 → bin 0, tasks at time 1 → bin 1, etc.

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → RCPSP).

        This direction is EXACT: all constraints can be represented in RCPSP.
        """
        return exact_transformation(
            use_cases=[
                "Exact encoding of bin packing as scheduling problem",
                "Incompatibility modeled via virtual unary resources",
                "Time slot t = bin t (tasks at same time go in same bin)",
                "All BinPack constraints preserved in RCPSP formulation",
            ]
        )

    def transform_problem(self, source_problem: BinPackProblem) -> RcpspProblem:
        """Transform BinPacking to RCPSP.

        Args:
            source_problem: BinPacking problem instance

        Returns:
            Equivalent RCPSP problem

        """
        bp = source_problem

        # Create resources
        resources = {
            "BinCapacity": bp.capacity_bin,  # Cumulative resource for bin capacity
        }

        # Add virtual resources for incompatible items
        incompatibility_map = {}  # (item_i, item_j) -> resource_name
        if bp.has_constraint:
            for i, j in bp.incompatible_items:
                # Create unique resource name (sorted to avoid duplicates)
                resource_name = f"Incompatible_{min(i, j)}_{max(i, j)}"
                resources[resource_name] = 1  # Capacity 1 (unary resource)
                incompatibility_map[(i, j)] = resource_name
                incompatibility_map[(j, i)] = resource_name  # Symmetric

        # Build mode_details: all tasks have single mode
        mode_details = {}

        # Source and sink dummy tasks
        source_task = "source"
        sink_task = "sink"

        mode_details[source_task] = {1: {"duration": 0, "BinCapacity": 0}}
        mode_details[sink_task] = {1: {"duration": 0, "BinCapacity": 0}}

        # Initialize incompatibility resources to 0 for dummy tasks
        for resource_name in resources:
            if resource_name.startswith("Incompatible"):
                mode_details[source_task][1][resource_name] = 0
                mode_details[sink_task][1][resource_name] = 0

        # Create task for each item
        for item in bp.list_items:
            task_name = f"item_{item.index}"

            mode = {
                "duration": 1,  # Unit duration
                "BinCapacity": int(item.weight),  # Resource requirement
            }

            # Initialize all incompatibility resources to 0
            for resource_name in resources:
                if resource_name.startswith("Incompatible"):
                    mode[resource_name] = 0

            # Set incompatibility resource consumption
            if bp.has_constraint:
                for other_item in bp.list_items:
                    if other_item.index != item.index:
                        key = (item.index, other_item.index)
                        if key in incompatibility_map:
                            resource_name = incompatibility_map[key]
                            mode[resource_name] = 1  # Consume the virtual resource

            mode_details[task_name] = {1: mode}

        # Build successors: no precedence, but all tasks must follow source
        successors = {}
        successors[source_task] = [f"item_{item.index}" for item in bp.list_items]
        successors[sink_task] = []

        for item in bp.list_items:
            task_name = f"item_{item.index}"
            successors[task_name] = [sink_task]

        # Horizon: worst case is one bin per item
        horizon = bp.nb_items

        return RcpspProblem(
            resources=resources,
            non_renewable_resources=[],
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            source_task=source_task,
            sink_task=sink_task,
        )

    def back_transform_solution(
        self, solution: RcpspSolution, source_problem: BinPackProblem
    ) -> BinPackSolution:
        """Transform RCPSP solution back to BinPacking solution.

        Args:
            solution: RCPSP solution
            source_problem: Original BinPacking problem

        Returns:
            Equivalent BinPacking solution

        """
        # Extract bin assignment from start times
        # Tasks scheduled at time t are assigned to bin t
        allocation = [0] * source_problem.nb_items

        for item in source_problem.list_items:
            task_name = f"item_{item.index}"
            if task_name in solution.rcpsp_schedule:
                start_time = solution.rcpsp_schedule[task_name]["start_time"]
                allocation[item.index] = start_time  # Bin = time slot

        return BinPackSolution(problem=source_problem, allocation=allocation)

    def forward_transform_solution(
        self, solution: BinPackSolution, target_problem: RcpspProblem
    ) -> Optional[RcpspSolution]:
        """Transform BinPacking solution to RCPSP solution (for warmstart).

        Args:
            solution: BinPacking solution
            target_problem: Target RCPSP problem

        Returns:
            Equivalent RCPSP solution for warmstart

        """
        # Build RCPSP schedule from bin allocation
        rcpsp_schedule = {
            "source": {"start_time": 0, "end_time": 0},
        }

        # Each item assigned to bin b is scheduled at time b
        for item in solution.problem.list_items:
            task_name = f"item_{item.index}"
            bin_id = solution.allocation[item.index]

            rcpsp_schedule[task_name] = {
                "start_time": bin_id,
                "end_time": bin_id + 1,  # Duration=1
            }

        # Sink starts after all tasks
        max_time = max(solution.allocation) + 1
        rcpsp_schedule["sink"] = {"start_time": max_time, "end_time": max_time}

        # All tasks use mode 1 (single mode)
        rcpsp_modes = [1] * len(solution.problem.list_items)

        return RcpspSolution(
            problem=target_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=rcpsp_modes,
        )
