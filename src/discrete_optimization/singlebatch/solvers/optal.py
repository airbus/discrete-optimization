import logging
from typing import Any, Optional

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.singlebatch.problem import (
    BatchProcessingSolution,
    SingleBatchProcessingProblem,
)

try:
    import optalcp as cp

    optalcp_available = True
except ImportError:
    cp = None
    optalcp_available = False

logger = logging.getLogger(__name__)


class OptalSingleBatchSolver(OptalCpSolver, WarmstartMixin):
    """OptalCP solver for the Single Batch-Processing Machine Scheduling Problem."""

    problem: SingleBatchProcessingProblem

    def __init__(
        self,
        problem: SingleBatchProcessingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        if not optalcp_available:
            raise RuntimeError(
                "OptalCP is not available. Install it from: https://www.optalcp.com/"
            )
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        problem = self.problem
        n = problem.nb_jobs
        cap = problem.capacity

        max_batches = n
        horizon = sum(j.processing_time for j in problem.jobs)
        max_p = max(j.processing_time for j in problem.jobs)

        # 1. Batch Intervals (Time Space)
        # Represents the actual start, duration, and end of the batches over time.
        self.batch_itvs = []
        for b in range(max_batches):
            iv = self.cp_model.interval_var(
                start=(0, horizon),
                length=(0, max_p),
                end=(0, horizon),
                optional=True,
                name=f"batch_{b}",
            )
            self.batch_itvs.append(iv)

        # Batches process sequentially on the single machine without gaps in indices
        for b in range(max_batches - 1):
            self.cp_model.enforce(
                self.cp_model.presence(self.batch_itvs[b])
                >= self.cp_model.presence(self.batch_itvs[b + 1])
            )
            self.cp_model.end_before_start(self.batch_itvs[b], self.batch_itvs[b + 1])
        self.cp_model._itv_presence_chain(self.batch_itvs)
        # 2. Dummy Intervals for Batch Assignment (Index Space)
        # Instead of time, the 'start' of these intervals represents the batch ID assigned to a job.
        self.dummy_assign_itvs = []
        self.batch_indices = []
        self.batch_indices_itv = []
        for j in range(n):
            iv = self.cp_model.interval_var(
                start=(0, max_batches - 1),
                length=1,
                end=(1, max_batches),
                optional=False,
                name=f"assign_job_{j}",
            )
            self.dummy_assign_itvs.append(iv)
            self.batch_indices.append(self.cp_model.start(iv))

        # 3. Capacity Constraint in Index Space
        # Creates a cumulative profile over the batch indices. No batch index can exceed capacity.
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(self.dummy_assign_itvs[j], problem.jobs[j].size)
                    for j in range(n)
                ]
            )
            <= cap
        )

        # 4. Job Intervals (Time Space) and Mapping
        # Represents the actual start, duration, and end of the jobs over time.
        self.job_itvs = []
        for j in range(n):
            # Crucially, the minimum length of the job interval is its processing time.
            iv = self.cp_model.interval_var(
                start=(0, horizon),
                length=(problem.jobs[j].processing_time, max_p),
                end=(0, horizon),
                optional=False,
                name=f"job_{j}",
            )
            self.job_itvs.append(iv)

        # Sync physical job time intervals with their assigned batch time intervals.
        # Because job length has a lower bound of `job.processing_time`, this implicitly forces
        # the batch interval length to be >= the processing time of all jobs mapped to it!
        self.cp_model._itv_mapping(self.job_itvs, self.batch_itvs, self.batch_indices)
        loads_per_batch = [
            self.cp_model.int_var(
                min=0, max=self.problem.capacity, name=f"loads_per_batch_{b}"
            )
            for b in range(max_batches)
        ]
        self.cp_model._pack(
            loads_per_batch,
            self.batch_indices,
            [problem.jobs[j].size for j in range(n)],
        )
        # 5. Objective
        makespan = self.cp_model.int_var(0, horizon, name="makespan")
        self.cp_model.enforce(
            self.cp_model.max([self.cp_model.end(iv) for iv in self.batch_itvs])
            == makespan
        )
        self.cp_model.minimize(makespan)

        self.variables = {
            "batch_itvs": self.batch_itvs,
            "dummy_assign_itvs": self.dummy_assign_itvs,
            "batch_indices": self.batch_indices,
            "job_itvs": self.job_itvs,
            "makespan": makespan,
        }

    def retrieve_solution(self, result: "cp.SolutionEvent") -> BatchProcessingSolution:
        """Constructs a DO solution from the OptalCP solver's internal state."""
        if result.solution is None:
            return self.problem.get_dummy_solution()

        solution = result.solution
        logger.info(f"Objective: {solution.get_objective()}")

        n = self.problem.nb_jobs
        job_to_batch = [-1] * n

        # Extract the batch index for each job
        for j in range(n):
            job_to_batch[j] = solution.get_start(self.variables["dummy_assign_itvs"][j])

        # BatchProcessingSolution auto-computes the physical schedule from the assignments!
        return BatchProcessingSolution(problem=self.problem, job_to_batch=job_to_batch)

    def set_warm_start(self, solution: BatchProcessingSolution) -> None:
        """Injects an initial solution into the OptalCP solver."""
        if self.cp_model is None:
            raise RuntimeError("Model must be initialized before setting warm start")

        warm_start = cp.Solution()
        n = self.problem.nb_jobs
        max_batches = n

        schedule = (
            solution.schedule_batch
        )  # Derived automatically by the Solution object
        used_batches = len(schedule)

        # 1. Provide hints for Real Time Batch Intervals
        for b in range(max_batches):
            if b < used_batches:
                start_time, end_time = schedule[b]
                warm_start.set_value(
                    self.variables["batch_itvs"][b], start_time, end_time
                )
            else:
                warm_start.set_absent(self.variables["batch_itvs"][b])

        # 2. Provide hints for Job to Batch Assignments & Real Time Job Intervals
        for j in range(n):
            alloc = solution.job_to_batch[j]

            # Index space hints
            warm_start.set_value(
                self.variables["dummy_assign_itvs"][j], alloc, alloc + 1
            )
            warm_start.set_value(self.variables["batch_indices"][j], alloc)

            # Time space hints
            start_time, end_time = schedule[alloc]
            warm_start.set_value(self.variables["job_itvs"][j], start_time, end_time)

        # 3. Provide hint for the Objective
        makespan_val = schedule[-1][1] if used_batches > 0 else 0
        warm_start.set_value(self.variables["makespan"], makespan_val)
        warm_start.set_objective(makespan_val)

        self.warm_start_solution = warm_start
        self.use_warm_start = True
