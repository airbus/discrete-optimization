#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections.abc import Hashable
from dataclasses import dataclass

from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    WithoutAllocationProblem,
    WithoutAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.batching import (
    BatchingProblem,
    BatchingSolution,
)
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    WithoutCalendarResourceProblem,
)
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    WithoutCumulativeResourceProblem,
)
from discrete_optimization.generic_tasks_tools.entities import VariableBatchEntity
from discrete_optimization.generic_tasks_tools.enums import AbsentValue, StartOrEnd
from discrete_optimization.generic_tasks_tools.exclusions import NoExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.multimode import (
    SinglemodeProblem,
    SinglemodeSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    WithoutNonRenewableResourceProblem,
)
from discrete_optimization.generic_tasks_tools.objectives.makespan import (
    MakespanObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.precedence import (
    WithoutPrecedenceProblem,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NoSkill,
    Skill,
    WithoutSkillProblem,
    WithoutSkillSolution,
)
from discrete_optimization.generic_tasks_tools.utils import optional_override_implem
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

logger = logging.getLogger(__name__)
Task = int
NonRenewableResource = str
NonSkillCumulativeResource = str


@dataclass
class Job:
    """Representation of a single job to be batched."""

    def __init__(self, job_id: int, processing_time: int, size: int):
        self.job_id = job_id
        self.processing_time = processing_time
        self.size = size

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, p={self.processing_time}, s={self.size})"


class BatchProcessingSolution(
    GenericSchedulingSolution[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        NoExclusionResource,
    ],
    WithoutSkillSolution[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutAllocationSolution[Task],
    SinglemodeSolution[Task],
    BatchingSolution[Task],
):
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
        # Build mapping from batch_id to schedule index
        self._build_batch_index_mapping()

    def _build_batch_index_mapping(self):
        """Build mapping from batch_id to index in schedule_batch."""
        used_batches = sorted(set(self.job_to_batch))
        self._batch_id_to_idx = {
            batch_id: idx for idx, batch_id in enumerate(used_batches)
        }

    def __setattr__(self, key, value):
        # Insure that we update the schedule after a job_to_batch change.
        super().__setattr__(key, value)
        if key == "job_to_batch":
            self.schedule_batch = self.problem.build_schedule_batch(self)
            self._build_batch_index_mapping()

    def get_end_time(self, task: Task) -> int | AbsentValue:
        """Get end time of a task (job)."""
        batch_id = self.job_to_batch[task]
        batch_idx = self._batch_id_to_idx[batch_id]
        return self.schedule_batch[batch_idx][1]

    def get_start_time(self, task: Task) -> int | AbsentValue:
        """Get start time of a task (job)."""
        batch_id = self.job_to_batch[task]
        batch_idx = self._batch_id_to_idx[batch_id]
        return self.schedule_batch[batch_idx][0]

    def is_present(self, task: Task) -> bool:
        """All tasks are present (not optional)."""
        return True

    # BatchingSolution interface
    def get_task_batch_assignment(self, task: Task) -> Hashable | None:
        """Get which batch a task is assigned to."""
        return self.job_to_batch[task]

    def copy(self) -> "BatchProcessingSolution":
        return BatchProcessingSolution(
            problem=self.problem,
            job_to_batch=list(self.job_to_batch),
            schedule_batch=list(self.schedule_batch) if self.schedule_batch else None,
        )

    def change_problem(self, new_problem: "SingleBatchProcessingProblem") -> None:
        self.problem = new_problem


class SingleBatchProcessingProblem(
    GenericSchedulingProblem[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        NoExclusionResource,
    ],
    WithoutSkillProblem[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    SinglemodeProblem[Task],
    WithoutCalendarResourceProblem[Task],
    WithoutCumulativeResourceProblem[Task, NoUnaryResource],
    WithoutNonRenewableResourceProblem[Task, NoUnaryResource],
    WithoutAllocationProblem[Task],
    WithoutPrecedenceProblem[Task],
    BatchingProblem[Task],
):
    """The Single Batch-Processing Machine Scheduling Problem.

    Jobs must be grouped into batches that execute sequentially.
    Each batch:
    - Has a capacity constraint (sum of job sizes <= capacity)
    - Executes for duration = max(processing_times of jobs in batch)
    - Batches execute in sequence (batch[i] finishes before batch[i+1] starts)
    """

    @optional_override_implem
    def get_entity_precedence_constraints(self):
        """Enforce temporal precedence: batch[i] must finish before batch[i+1] starts."""
        succ = {}
        batch_entities = self.get_batch_entities()
        for i in range(len(batch_entities) - 1):
            succ[batch_entities[i]] = [batch_entities[i + 1]]
        return succ

    @optional_override_implem
    def get_batch_presence_chains(self) -> list[list[int]]:
        """Enforce presence ordering: if batch[i+1] is used, batch[i] must be used.

        This is a symmetry-breaking constraint for single-machine batching.
        """
        return [self.get_batch_ids()]  # All batches in one chain

    def get_list_objective_computer(self) -> list[ObjectiveComputer]:
        return [MakespanObjectiveComputer(self, weight_objective=1)]

    @property
    def non_skill_cumulative_resources_list(self) -> list[Skill]:
        return []

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.jobs[task].processing_time

    def __init__(self, jobs: list[Job], capacity: int, max_batches: int | None = None):
        """Initialize problem.

        Args:
            jobs: List of jobs to schedule
            capacity: Maximum capacity (sum of sizes) per batch
            max_batches: Maximum number of batches (default: len(jobs))
        """
        self.jobs = jobs
        self.capacity = capacity
        self.nb_jobs = len(jobs)
        self.max_batches = max_batches if max_batches is not None else self.nb_jobs

        # Create batch entities (all jobs can potentially go into any batch)
        candidate_tasks = frozenset(range(self.nb_jobs))
        self._batch_entities = [
            VariableBatchEntity(
                batch_id=batch_id,
                candidate_tasks=candidate_tasks,
            )
            for batch_id in range(self.max_batches)
        ]

    # =========================================================================
    # TasksProblem interface
    # =========================================================================

    @property
    def tasks_list(self) -> list[Task]:
        """Return list of all tasks (jobs)."""
        return list(range(self.nb_jobs))

    def is_optional(self, task: Task) -> bool:
        """All jobs must be scheduled."""
        return False

    def get_solution_type(self) -> type[Solution]:
        return BatchProcessingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                Objective.MAKESPAN: ObjectiveDoc(
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

    # =========================================================================
    # SchedulingProblem interface
    # =========================================================================

    def get_task_duration(self, task: Task, mode: int = 1) -> int:
        """Return processing time of a job."""
        return self.jobs[task].processing_time

    def get_makespan_upper_bound(self) -> int:
        """Upper bound on makespan (all jobs run sequentially)."""
        return sum(job.processing_time for job in self.jobs)

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Lower bound on task start/end time."""
        return 0

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Upper bound on task start/end time."""
        return self.get_makespan_upper_bound()

    # =========================================================================
    # BatchingProblem interface
    # =========================================================================

    def get_batch_entities(self) -> list[VariableBatchEntity[Task]]:
        """Return all batch entities."""
        return self._batch_entities

    def get_candidate_tasks_for_batch(self, batch_id: int) -> frozenset[Task]:
        """All jobs can go into any batch."""
        return frozenset(range(self.nb_jobs))

    def get_task_size(self, task: Task) -> int:
        """Return size of a job (for capacity constraint)."""
        return self.jobs[task].size

    def get_batch_capacity(self, batch_id: int) -> int:
        """Return capacity of a batch."""
        return self.capacity

    def is_batch_optional(self, batch_id: int) -> bool:
        """Batches can be empty (unused)."""
        return True

    def must_be_batched(self, task: Task) -> bool:
        """All jobs must be assigned to a batch."""
        return True

    def can_be_batched(self, task: Task) -> bool:
        """All jobs can be batched."""
        return True

    def get_batch_duration_bounds(
        self, batch_id: int, tasks: frozenset[Task]
    ) -> tuple[int | None, int | None]:
        """Batch duration is max of job processing times."""
        if not tasks:
            return 0, 0
        max_duration = max(self.jobs[task].processing_time for task in tasks)
        return max_duration, max_duration

    # =========================================================================
    # Evaluation and solution building
    # =========================================================================

    def evaluate(self, variable: BatchProcessingSolution) -> dict[str, float]:
        """Calculate the Makespan (Cmax) of the current batching solution."""
        violation = self.compute_batches_violation(variable)
        return {
            Objective.MAKESPAN: variable.schedule_batch[-1][1],
            "violation": violation,
        }

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
        batch_duration: dict[int, int] = {}
        for job_idx, batch_id in enumerate(variable.job_to_batch):
            job = self.jobs[job_idx]
            batch_sizes[batch_id] = batch_sizes.get(batch_id, 0) + job.size
            batch_duration[batch_id] = max(
                batch_duration.get(batch_id, 0), job.processing_time
            )
            if batch_sizes[batch_id] > self.capacity:
                return False
        sorted_batches = sorted(batch_sizes)
        for i in range(len(variable.schedule_batch)):
            if i >= 1:
                if not (
                    variable.schedule_batch[i][0] >= variable.schedule_batch[i - 1][1]
                ):
                    logger.info(f"Precedence between batches {i - 1}/{i} not respected")
                    return False
            if (
                variable.schedule_batch[i][1] - variable.schedule_batch[i][0]
                < batch_duration[sorted_batches[i]]
            ):
                logger.info(f"Duration of batch {i} lower than needed")
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
