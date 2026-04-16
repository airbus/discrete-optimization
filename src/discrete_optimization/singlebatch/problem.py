#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import ListInteger

Task = int


@dataclass
class Job:
    """Representation of a single job to be batched."""

    def __init__(self, job_id: int, processing_time: int, size: int):
        self.job_id = job_id
        self.processing_time = processing_time
        self.size = size

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, p={self.processing_time}, s={self.size})"


class BatchProcessingSolution(SchedulingSolution[Task]):
    """A solution mapping jobs to distinct batches."""

    problem: "SingleBatchProcessingProblem"

    def __init__(
        self,
        problem: "SingleBatchProcessingProblem",
        job_to_batch: list[int],
        schedule_batch: list[tuple[int, int]] = None,
    ):
        super().__init__(problem)
        self.job_to_batch = job_to_batch
        self.schedule_batch = schedule_batch
        if self.schedule_batch is None:
            self.schedule_batch = self.problem.build_schedule_batch(self)

    def __setattr__(self, key, value):
        # Insure that we update the schedule after a job_to_batch change.
        super().__setattr__(key, value)
        if key == "job_to_batch":
            self.schedule_batch = self.problem.build_schedule_batch(self)

    def get_end_time(self, task: Task) -> int:
        return self.schedule_batch[self.job_to_batch[task]][1]

    def get_start_time(self, task: Task) -> int:
        return self.schedule_batch[self.job_to_batch[task]][0]

    def copy(self) -> "BatchProcessingSolution":
        return BatchProcessingSolution(
            problem=self.problem,
            job_to_batch=list(self.job_to_batch),
            schedule_batch=self.schedule_batch,
        )

    def change_problem(self, new_problem: "SingleBatchProcessingProblem") -> None:
        self.problem = new_problem


class SingleBatchProcessingProblem(SchedulingProblem[Task]):
    """The Single Batch-Processing Machine Scheduling Problem."""

    def get_makespan_upper_bound(self) -> int:
        return sum([j.processing_time for j in self.jobs])

    @property
    def tasks_list(self) -> list[Task]:
        return list(range(self.nb_jobs))

    def get_solution_type(self) -> type[Solution]:
        return BatchProcessingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "violation": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=1000
                ),
            },
        )

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "job_to_batch": ListInteger(
                    lows=0, ups=self.nb_jobs - 1, length=self.nb_jobs
                )
            }
        )

    def __init__(self, jobs: list[Job], capacity: int):
        self.jobs = jobs
        self.capacity = capacity
        self.nb_jobs = len(jobs)

    def evaluate(self, variable: BatchProcessingSolution) -> dict[str, float]:
        """Calculate the Makespan (Cmax) of the current batching solution."""
        violation = self.compute_batches_violation(variable)
        return {"makespan": variable.schedule_batch[-1][1], "violation": violation}

    def compute_processing_time_batch(
        self, variable: BatchProcessingSolution
    ) -> dict[int, int]:
        batch_processing_times: dict[int, int] = {}
        for job_idx, batch_id in enumerate(variable.job_to_batch):
            job = self.jobs[job_idx]
            # The processing time of a batch is the max of the processing times of jobs within it
            if batch_id not in batch_processing_times:
                batch_processing_times[batch_id] = job.processing_time
            else:
                batch_processing_times[batch_id] = max(
                    batch_processing_times[batch_id], job.processing_time
                )
        return batch_processing_times

    def build_schedule_batch(
        self, variable: BatchProcessingSolution
    ) -> list[tuple[int, int]]:
        batch_processing_times = self.compute_processing_time_batch(variable)
        schedule = []
        cur_time = 0
        for b in sorted(batch_processing_times):
            schedule.append((cur_time, cur_time + batch_processing_times[b]))
            cur_time = schedule[-1][1]
        return schedule

    def satisfy(self, variable: BatchProcessingSolution) -> bool:
        """Check if all capacity constraints are respected."""
        batch_sizes: dict[int, int] = {}

        for job_idx, batch_id in enumerate(variable.job_to_batch):
            job = self.jobs[job_idx]
            batch_sizes[batch_id] = batch_sizes.get(batch_id, 0) + job.size
            if batch_sizes[batch_id] > self.capacity:
                return False
        return True

    def compute_batches_violation(self, variable: BatchProcessingSolution) -> int:
        """Returns sum of all batch processing violations."""
        batch_sizes: dict[int, int] = {}
        for job_idx, batch_id in enumerate(variable.job_to_batch):
            job = self.jobs[job_idx]
            batch_sizes[batch_id] = batch_sizes.get(batch_id, 0) + job.size
        return sum(max(batch_sizes[b] - self.capacity, 0) for b in batch_sizes)

    def get_dummy_solution(self) -> BatchProcessingSolution:
        """Create a trivial valid solution (one job per batch)."""
        job_to_batch = [i for i in range(self.nb_jobs)]
        return BatchProcessingSolution(self, job_to_batch)
