#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from SingleBatch to OvenSched.

SingleBatch is a special case of OvenSched with:
- 1 machine
- 1 attribute (all jobs have same type)
- No setup times/costs
- No time windows
- Fixed processing times (min_duration = max_duration)
"""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    subset_transformation,
)
from discrete_optimization.ovensched.problem import (
    MachineData,
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
    TaskData,
)
from discrete_optimization.singlebatch.problem import (
    BatchProcessingSolution,
    SingleBatchProcessingProblem,
)


class SinglebatchToOvenschedTransformation(
    ProblemTransformation[
        SingleBatchProcessingProblem,
        BatchProcessingSolution,
        OvenSchedulingProblem,
        OvenSchedulingSolution,
    ]
):
    """Transform SingleBatch problem to OvenSched problem.

    SingleBatch is a special case of OvenSched, so this is a SUBSET transformation (exact).

    Mapping:
    - Single machine → 1 machine in OvenSched
    - Jobs → Tasks with:
      - Single attribute (all tasks have attribute=0)
      - min_duration = max_duration = processing_time
      - size → size
      - earliest_start = 0
      - latest_end = large value (no deadline)
      - eligible_machines = {0} (only the single machine)
    - Capacity → machine capacity
    - No setup times → all setup_times[i][j] = 0
    - No setup costs → all setup_costs[i][j] = 0
    - Machine availability → [(0, large_value)] (always available)

    This is an EXACT transformation:
    - Forward: Every SingleBatch problem is a valid OvenSched problem
    - Backward: Solutions map directly (batch assignments are preserved)
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (SingleBatch → OvenSched).

        This is a SUBSET transformation: SingleBatch ⊂ OvenSched.
        """
        return subset_transformation(
            use_cases=[
                "Use OvenSched solvers on SingleBatch problems",
                "Benchmark OvenSched solvers on simpler instances",
                "Prototype on SingleBatch before tackling full OvenSched",
            ],
            assumptions=[
                "SingleBatch is a special case of OvenSched with restrictions",
                "1 machine, 1 attribute, no setup times/costs, no time windows",
            ],
        )

    def transform_problem(
        self, source_problem: SingleBatchProcessingProblem
    ) -> OvenSchedulingProblem:
        """Transform SingleBatch to OvenSched.

        Args:
            source_problem: SingleBatch problem instance

        Returns:
            Equivalent OvenSched problem with 1 machine and 1 attribute
        """
        n_jobs = source_problem.nb_jobs
        n_machines = 1  # Single machine

        # All tasks have the same attribute (0)
        single_attribute = 0

        # Large value for unbounded time windows
        large_time = source_problem.get_makespan_upper_bound() * 2

        # Create task data
        tasks_data = []
        for i, job in enumerate(source_problem.jobs):
            tasks_data.append(
                TaskData(
                    attribute=single_attribute,
                    min_duration=job.processing_time,
                    max_duration=job.processing_time,
                    earliest_start=0,
                    latest_end=large_time,
                    eligible_machines={0},  # Only machine 0
                    size=job.size,
                )
            )

        # Create machine data
        machines_data = [
            MachineData(
                capacity=source_problem.capacity,
                initial_attribute=single_attribute,
                availability=[(0, large_time)],  # Always available
            )
        ]

        # Setup times and costs (all zeros since no setup)
        # Need 1x1 matrix for the single attribute
        num_attributes = 1
        setup_times = [[0] * num_attributes for _ in range(num_attributes)]
        setup_costs = [[0] * num_attributes for _ in range(num_attributes)]

        return OvenSchedulingProblem(
            n_jobs=n_jobs,
            n_machines=n_machines,
            tasks_data=tasks_data,
            machines_data=machines_data,
            setup_costs=setup_costs,
            setup_times=setup_times,
        )

    def back_transform_solution(
        self,
        solution: OvenSchedulingSolution,
        source_problem: SingleBatchProcessingProblem,
    ) -> BatchProcessingSolution:
        """Transform OvenSched solution back to SingleBatch solution.

        Args:
            solution: OvenSched solution
            source_problem: Original SingleBatch problem

        Returns:
            Equivalent SingleBatch solution
        """
        # Extract batches from machine 0
        machine_0_schedule = solution.schedule_per_machine.get(0, [])

        # Create job_to_batch mapping
        job_to_batch = [0] * source_problem.nb_jobs

        for batch_idx, schedule_info in enumerate(machine_0_schedule):
            for task_id in schedule_info.tasks:
                job_to_batch[task_id] = batch_idx

        # Create schedule_batch for SingleBatch
        schedule_batch = []
        for schedule_info in machine_0_schedule:
            schedule_batch.append((schedule_info.start_time, schedule_info.end_time))

        return BatchProcessingSolution(
            problem=source_problem,
            job_to_batch=job_to_batch,
            schedule_batch=schedule_batch,
        )

    def forward_transform_solution(
        self,
        solution: BatchProcessingSolution,
        target_problem: OvenSchedulingProblem,
    ) -> Optional[OvenSchedulingSolution]:
        """Transform SingleBatch solution to OvenSched solution (for warmstart).

        Args:
            solution: SingleBatch solution
            target_problem: Target OvenSched problem

        Returns:
            Equivalent OvenSched solution for warmstart
        """
        # Group jobs by batch
        batches_dict = {}
        for job_idx, batch_id in enumerate(solution.job_to_batch):
            if batch_id not in batches_dict:
                batches_dict[batch_id] = set()
            batches_dict[batch_id].add(job_idx)

        # Create schedule for machine 0
        schedule_infos = []
        for batch_id in sorted(batches_dict.keys()):
            tasks_in_batch = batches_dict[batch_id]

            # Get batch timing from solution
            if batch_id < len(solution.schedule_batch):
                start_time, end_time = solution.schedule_batch[batch_id]
            else:
                # Fallback if schedule_batch not available
                start_time = 0
                end_time = max(
                    solution.problem.jobs[t].processing_time for t in tasks_in_batch
                )

            schedule_infos.append(
                ScheduleInfo(
                    tasks=tasks_in_batch,
                    task_attribute=0,  # Single attribute
                    start_time=start_time,
                    end_time=end_time,
                    machine_batch_index=(0, batch_id),
                )
            )

        schedule_per_machine = {0: schedule_infos}

        return OvenSchedulingSolution(
            problem=target_problem,
            schedule_per_machine=schedule_per_machine,
        )
