#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from SALBP to RCALBP_L (Resource-Constrained Assembly Line Balancing with Learning)."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    subset_transformation,
)

from discrete_optimization.alb.rcalbp_l.problem import (
    RCALBPLProblem,
    RCALBPLSolution,
)
from discrete_optimization.alb.salbp.problem import SalbpProblem, SalbpSolution


class SalbpToRcalbpLTransformation(
    ProblemTransformation[SalbpProblem, SalbpSolution, RCALBPLProblem, RCALBPLSolution]
):
    """Transform SALBP to RCALBP_L.

    Mapping:
    - SALBP tasks → RCALBP_L tasks (single period, no learning)
    - SALBP stations → RCALBP_L workstations
    - SALBP cycle time → RCALBP_L target cycle time
    - SALBP precedence → RCALBP_L precedence
    - No resources, no zones (empty sets)
    - Single period (no learning effect)
    - Task durations remain constant across workstations

    This is a SUBSET transformation: SALBP is a special case of RCALBP_L where:
    - nb_resources = 0
    - nb_zones = 0
    - nb_periods = 1
    - No learning effect (durations constant)
    """

    def __init__(self, nb_stations_upper_bound: Optional[int] = None):
        """Initialize transformation.

        Args:
            nb_stations_upper_bound: Upper bound on number of stations
                (default: number of tasks)

        """
        self.nb_stations_upper_bound = nb_stations_upper_bound

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward transformation (SALBP → RCALBP_L)."""
        return subset_transformation(
            use_cases=[
                "Use RCALBP_L solvers for SALBP problems",
                "SALBP is a special case of RCALBP_L (no resources, single period)",
                "Benchmark RCALBP_L algorithms on simpler SALBP instances",
            ],
            assumptions=[
                "No resource constraints",
                "No zone constraints",
                "Single period (no learning effect)",
                "Durations constant across workstations",
            ],
        )

    def transform_problem(self, source_problem: SalbpProblem) -> RCALBPLProblem:
        """Transform SALBP to RCALBP_L.

        Args:
            source_problem: SALBP problem instance

        Returns:
            Equivalent RCALBP_L problem (no resources, single period)

        """
        # Parameters
        nb_tasks = source_problem.nb_tasks
        nb_stations = (
            self.nb_stations_upper_bound
            if self.nb_stations_upper_bound is not None
            else nb_tasks
        )
        nb_periods = 1  # Single period (no learning)
        nb_resources = 1  # No resources
        nb_zones = 0  # No zones

        # Cycle time constraints
        c_target = source_problem.cycle_time
        c_max = source_problem.cycle_time  # Same as target for SALBP

        # Precedence constraints
        # SALBP precedence: list of (pred, succ) pairs
        # RCALBP_L precedence: list of ((pred, period), (succ, period)) pairs
        # Since we have single period, all tasks are in period 0
        precedences = [
            (source_problem.tasks_to_index[pred], source_problem.tasks_to_index[succ])
            for pred, succ in source_problem.precedence
        ]

        # Durations: RCALBP_L uses durations[task][experience_level]
        # For SALBP, no learning effect, so duration is constant
        # durations[t][0] = task time for task t (no experience)
        durations = [
            [source_problem.task_times[task]] * nb_stations
            for task in source_problem.tasks
        ]

        # Empty resource/zone arrays
        capa_resources = [1]
        cons_resources = [[1 for i in range(nb_tasks)]]
        capa_zones = []
        cons_zones = []
        neutr_zones = [[] for i in range(nb_tasks)]

        return RCALBPLProblem(
            c_target=c_target,
            c_max=c_max,
            nb_stations=nb_stations,
            nb_periods=nb_periods,
            nb_tasks=nb_tasks,
            precedences=precedences,
            durations=durations,
            nb_resources=nb_resources,
            capa_resources=capa_resources,
            cons_resources=cons_resources,
            nb_zones=nb_zones,
            capa_zones=capa_zones,
            cons_zones=cons_zones,
            neutr_zones=neutr_zones,
            p_start=nb_stations - 1,
            p_end=nb_stations,  # Single period [0, 1)
        )

    def back_transform_solution(
        self, solution: RCALBPLSolution, source_problem: SalbpProblem
    ) -> SalbpSolution:
        """Transform RCALBP_L solution back to SALBP solution.

        Args:
            solution: RCALBP_L solution
            source_problem: Original SALBP problem

        Returns:
            Equivalent SALBP solution

        """
        # Extract workstation assignments from RCALBP_L solution
        # RCALBP_L: wks[task_id] = workstation
        # SALBP: allocation_to_station[task_index] = station

        # Map task IDs to indices
        allocation_to_station = [
            solution.wks[source_problem.tasks_to_index[task]]
            for task in source_problem.tasks
        ]

        return SalbpSolution(
            problem=source_problem,
            allocation_to_station=allocation_to_station,
        )

    def forward_transform_solution(
        self, solution: SalbpSolution, target_problem: RCALBPLProblem
    ) -> Optional[RCALBPLSolution]:
        """Transform SALBP solution to RCALBP_L solution (for warmstart).

        Args:
            solution: SALBP solution
            target_problem: Target RCALBP_L problem

        Returns:
            Equivalent RCALBP_L solution

        """
        # Build workstation assignments
        # SALBP: allocation_to_station[task_index] = station
        # RCALBP_L: wks[task_id] = workstation
        wks = {
            target_problem.tasks[i]: solution.allocation_to_station[i]
            for i in range(len(solution.allocation_to_station))
        }

        # Empty resource allocation (no resources)
        raw = {}

        # Build schedule using RCALBP_L's scheduling algorithm
        # We use a simple greedy schedule for period 0
        # Target starts: use station assignment as priority
        target_starts = {task: wks[task] for task in wks}

        # Build full solution using RCALBP_L's built-in scheduler
        return target_problem.build_full_solution(
            wks=wks,
            raw=raw,
            target_starts=target_starts,
        )
