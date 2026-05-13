#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from FlexibleJobShop to RCPSP."""

from typing import Optional

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution


class FjspToRcpspTransformation(
    ProblemTransformation[
        FJobShopProblem, FJobShopSolution, RcpspProblem, RcpspSolution
    ]
):
    """Transform FlexibleJobShop to RCPSP.

    Mapping:
    - (job_j, subjob_k) → task_{j}_{k}
    - Machine options → modes for each task
    - Machines → renewable resources (capacity 1)
    - Processing time on machine → duration for that mode
    - Job precedence → task successors

    This allows using RCPSP solvers for flexible job shop problems.
    """

    def __init__(self):
        """Initialize transformation."""
        # We'll store mapping between (job, subjob) and task names
        self.task_to_tuple: dict[str, tuple[int, int]] = {}
        self.tuple_to_task: dict[tuple[int, int], str] = {}

    def _make_task_name(self, job: int, subjob: int) -> str:
        """Create task name from (job, subjob) indices."""
        return f"task_{job}_{subjob}"

    def transform_problem(self, source_problem: FJobShopProblem) -> RcpspProblem:
        """Transform FlexibleJobShop to RCPSP.

        Args:
            source_problem: FlexibleJobShop problem instance

        Returns:
            Equivalent RCPSP problem

        """
        # Build task name mappings
        self.task_to_tuple = {}
        self.tuple_to_task = {}

        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx in range(len(job.sub_jobs)):
                task_name = self._make_task_name(job_idx, subjob_idx)
                self.task_to_tuple[task_name] = (job_idx, subjob_idx)
                self.tuple_to_task[(job_idx, subjob_idx)] = task_name

        # Create resources: one per machine (unary resources)
        resources = {f"M{machine}": 1 for machine in range(source_problem.n_machines)}

        # Build mode_details: task -> (mode -> resource requirements + duration)
        mode_details = {}

        # Add source and sink dummy tasks
        source_task = "source"
        sink_task = "sink"

        mode_details[source_task] = {1: {"duration": 0}}
        mode_details[sink_task] = {1: {"duration": 0}}

        # Initialize resource requirements to 0 for all resources
        for resource in resources:
            mode_details[source_task][1][resource] = 0
            mode_details[sink_task][1][resource] = 0

        # Build mode_details for each task
        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx, subjob_options in enumerate(job.sub_jobs):
                task_name = self._make_task_name(job_idx, subjob_idx)
                mode_details[task_name] = {}

                # Each machine option becomes a mode
                for mode_idx, subjob in enumerate(subjob_options, start=1):
                    mode = {
                        "duration": subjob.processing_time,
                    }

                    # Initialize all machines to 0
                    for machine in range(source_problem.n_machines):
                        mode[f"M{machine}"] = 0

                    # Set the required machine to 1
                    mode[f"M{subjob.machine_id}"] = 1

                    mode_details[task_name][mode_idx] = mode

        # Build successors (precedence constraints)
        successors = {source_task: [], sink_task: []}

        # Add all first subjobs as successors of source
        for job_idx, job in enumerate(source_problem.list_jobs):
            if len(job.sub_jobs) > 0:
                first_task = self._make_task_name(job_idx, 0)
                successors[source_task].append(first_task)

        # Add precedence within each job
        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx in range(len(job.sub_jobs)):
                task_name = self._make_task_name(job_idx, subjob_idx)

                if subjob_idx + 1 < len(job.sub_jobs):
                    # Has a successor within the same job
                    next_task = self._make_task_name(job_idx, subjob_idx + 1)
                    successors[task_name] = [next_task]
                else:
                    # Last subjob of the job → leads to sink
                    successors[task_name] = [sink_task]

        return RcpspProblem(
            resources=resources,
            non_renewable_resources=[],
            mode_details=mode_details,
            successors=successors,
            horizon=source_problem.horizon,
            source_task=source_task,
            sink_task=sink_task,
        )

    def back_transform_solution(
        self, solution: RcpspSolution, source_problem: FJobShopProblem
    ) -> FJobShopSolution:
        """Transform RCPSP solution back to FlexibleJobShop solution.

        Args:
            solution: RCPSP solution
            source_problem: Original FlexibleJobShop problem

        Returns:
            Equivalent FlexibleJobShop solution

        """
        # Build FJSP schedule from RCPSP schedule
        fjsp_schedule = [[None] * len(job.sub_jobs) for job in source_problem.list_jobs]

        for task_name, task_details in solution.rcpsp_schedule.items():
            if task_name in ["source", "sink"]:
                continue

            job_idx, subjob_idx = self.task_to_tuple[task_name]
            start = task_details["start_time"]
            end = task_details["end_time"]

            # Get mode (which machine option was chosen)
            # rcpsp_modes is indexed by tasks_list_non_dummy (excluding source/sink)
            task_idx_non_dummy = solution.problem.index_task_non_dummy[task_name]
            mode = solution.rcpsp_modes[task_idx_non_dummy]
            option = mode - 1  # RCPSP modes are 1-indexed, FJSP options are 0-indexed

            # Get machine from the chosen option
            machine_id = (
                source_problem.list_jobs[job_idx]
                .sub_jobs[subjob_idx][option]
                .machine_id
            )

            fjsp_schedule[job_idx][subjob_idx] = (start, end, machine_id, option)

        return FJobShopSolution(problem=source_problem, schedule=fjsp_schedule)

    def forward_transform_solution(
        self, solution: FJobShopSolution, target_problem: RcpspProblem
    ) -> Optional[RcpspSolution]:
        """Transform FlexibleJobShop solution to RCPSP solution (for warmstart).

        Args:
            solution: FlexibleJobShop solution
            target_problem: Target RCPSP problem

        Returns:
            Equivalent RCPSP solution for warmstart

        """
        # Build RCPSP schedule from FJSP schedule
        rcpsp_schedule = {
            "source": {"start_time": 0, "end_time": 0},
            "sink": {
                "start_time": solution.get_max_end_time(),
                "end_time": solution.get_max_end_time(),
            },
        }

        rcpsp_modes = [1, 1]  # source and sink use mode 1

        # Convert FJSP schedule to RCPSP schedule
        for job_idx, job_schedule in enumerate(solution.schedule):
            for subjob_idx, (start, end, machine_id, option) in enumerate(job_schedule):
                task_name = self._make_task_name(job_idx, subjob_idx)

                rcpsp_schedule[task_name] = {
                    "start_time": start,
                    "end_time": end,
                }

                # Mode in RCPSP is 1-indexed (option + 1)
                mode = option + 1
                rcpsp_modes.append(mode)

        return RcpspSolution(
            problem=target_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=rcpsp_modes,
        )
