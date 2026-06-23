#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from CommonShopProblem (JSP/FJSP/OSP) to multimode RCPSP."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem


class ShopToRcpspMultimodeTransformation(
    ProblemTransformation[
        CommonShopProblem, AnyShopSolution, RcpspProblem, RcpspSolution
    ]
):
    """Transform CommonShopProblem to multimode RCPSP.

    This transformation works for JSP, FJSP, and OSP problems:
    - JSP: Single mode per task (one recipe per subjob)
    - FJSP: Multiple modes per task (multiple recipe options per subjob)
    - OSP: Single mode per task, no precedence constraints

    Mapping:
    - (job_j, subjob_k) → task_{j}_{k}
    - Recipe options → modes for each task
    - Machines → renewable resources (capacity 1)
    - Processing time on machine → duration for that mode
    - Job precedence + no-overlap constraints → task successors and constraints

    """

    def __init__(self):
        """Initialize transformation."""
        self.task_to_tuple: dict[str, tuple[int, int]] = {}
        self.tuple_to_task: dict[tuple[int, int], str] = {}

    def _make_task_name(self, job: int, subjob: int) -> str:
        """Create task name from (job, subjob) indices."""
        return f"task_{job}_{subjob}"

    def transform_problem(self, source_problem: CommonShopProblem) -> RcpspProblem:
        """Transform CommonShopProblem to multimode RCPSP.

        Args:
            source_problem: CommonShopProblem instance (JSP/FJSP/OSP)

        Returns:
            Equivalent multimode RCPSP problem

        """
        # Build task name mappings
        self.task_to_tuple = {}
        self.tuple_to_task = {}

        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx in range(source_problem.nb_subjob_per_job[job_idx]):
                task_name = self._make_task_name(job_idx, subjob_idx)
                self.task_to_tuple[task_name] = (job_idx, subjob_idx)
                self.tuple_to_task[(job_idx, subjob_idx)] = task_name

        # Create resources: one per machine (unary resources)
        resources = {f"M{machine}": 1 for machine in range(source_problem.n_machines)}

        # Get no-overlap sets (tasks within same job must not overlap)
        no_overlap_sets = source_problem.get_no_overlap()

        # Create a renewable resource for each no-overlap set
        # Each resource has capacity 1, forcing tasks in the set to not overlap
        no_overlap_resources = {}
        for idx, task_set in enumerate(no_overlap_sets):
            resource_name = f"NoOverlap_{idx}"
            resources[resource_name] = 1
            no_overlap_resources[resource_name] = task_set

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
            for subjob_idx, subjob in enumerate(job.subjobs):
                task_name = self._make_task_name(job_idx, subjob_idx)
                task_tuple = (job_idx, subjob_idx)
                mode_details[task_name] = {}

                # Each recipe option becomes a mode
                for mode_idx, recipe in enumerate(subjob.recipes, start=1):
                    mode = {
                        "duration": recipe.processing_time,
                    }

                    # Initialize all machines to 0
                    for machine in range(source_problem.n_machines):
                        mode[f"M{machine}"] = 0

                    # Set the required machine to 1
                    mode[f"M{recipe.machine_index}"] = 1

                    # Add no-overlap resource consumption
                    # If this task belongs to a no-overlap set, it consumes the corresponding resource
                    for resource_name, task_set in no_overlap_resources.items():
                        if task_tuple in task_set:
                            mode[resource_name] = 1
                        else:
                            mode[resource_name] = 0

                    mode_details[task_name][mode_idx] = mode

        # Build successors (precedence constraints)
        successors = {source_task: [], sink_task: []}

        # Get precedence constraints from the problem
        precedence = source_problem.get_precedence_constraints()

        # First, collect all first tasks of each job
        first_tasks_of_jobs = set()
        for job_idx, job in enumerate(source_problem.list_jobs):
            if len(job.subjobs) > 0:
                first_task = self._make_task_name(job_idx, 0)
                first_tasks_of_jobs.add(first_task)
                successors[source_task].append(first_task)

        # Build precedence from the problem's get_precedence_constraints
        for task, task_successors in precedence.items():
            task_name = self.tuple_to_task[task]
            if task_name not in successors:
                successors[task_name] = []

            for succ_task in task_successors:
                succ_task_name = self.tuple_to_task[succ_task]
                successors[task_name].append(succ_task_name)

        # Handle tasks with no successors -> they lead to sink
        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx in range(len(job.subjobs)):
                task_name = self._make_task_name(job_idx, subjob_idx)
                if task_name not in successors:
                    successors[task_name] = []

                # Check if this is a last task (no successors in precedence)
                task_tuple = (job_idx, subjob_idx)
                has_successor = False
                if task_tuple in precedence:
                    if len(precedence[task_tuple]) > 0:
                        has_successor = True

                if not has_successor:
                    # This is a terminal task, it should lead to sink
                    successors[task_name].append(sink_task)

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
        self, solution: RcpspSolution, source_problem: CommonShopProblem
    ) -> AnyShopSolution:
        """Transform RCPSP solution back to CommonShopProblem solution.

        Args:
            solution: RCPSP solution
            source_problem: Original CommonShopProblem

        Returns:
            Equivalent CommonShopProblem solution

        """
        # Build shop schedule from RCPSP schedule
        shop_schedule = [[None] * len(job.subjobs) for job in source_problem.list_jobs]
        machine_index = [[None] * len(job.subjobs) for job in source_problem.list_jobs]

        for task_name, task_details in solution.rcpsp_schedule.items():
            if task_name in ["source", "sink"]:
                continue

            job_idx, subjob_idx = self.task_to_tuple[task_name]
            start = task_details["start_time"]
            end = task_details["end_time"]

            # Get mode (which recipe option was chosen)
            task_idx_non_dummy = solution.problem.index_task_non_dummy[task_name]
            mode = solution.rcpsp_modes[task_idx_non_dummy]
            option = mode - 1  # RCPSP modes are 1-indexed, shop options are 0-indexed

            # Get machine from the chosen option
            machine_id = (
                source_problem.list_jobs[job_idx]
                .subjobs[subjob_idx]
                .recipes[option]
                .machine_index
            )

            shop_schedule[job_idx][subjob_idx] = (start, end)
            machine_index[job_idx][subjob_idx] = machine_id

        return AnyShopSolution(
            problem=source_problem,
            schedule=shop_schedule,
            machine_index=machine_index,
        )

    def forward_transform_solution(
        self, solution: AnyShopSolution, target_problem: RcpspProblem
    ) -> Optional[RcpspSolution]:
        """Transform CommonShopProblem solution to RCPSP solution (for warm-start).

        Args:
            solution: CommonShopProblem solution
            target_problem: Target RCPSP problem

        Returns:
            Equivalent RCPSP solution for warm-start

        """
        # Build RCPSP schedule from shop schedule
        rcpsp_schedule = {
            "source": {"start_time": 0, "end_time": 0},
            "sink": {
                "start_time": solution.get_max_end_time(),
                "end_time": solution.get_max_end_time(),
            },
        }

        rcpsp_modes = [1, 1]  # source and sink use mode 1

        # Convert shop schedule to RCPSP schedule
        for job_idx, job_schedule in enumerate(solution.schedule):
            for subjob_idx, (start, end) in enumerate(job_schedule):
                task_name = self._make_task_name(job_idx, subjob_idx)
                option = solution.get_mode((job_idx, subjob_idx))
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
