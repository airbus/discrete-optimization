import logging
from enum import Enum
from typing import Any, Optional

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, LinearExpr

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.singlebatch.problem import (
    BatchProcessingSolution,
    SingleBatchProcessingProblem,
)

logger = logging.getLogger(__name__)


class ModelingBpm(Enum):
    BINARY = 0
    SCHEDULING = 1


class CpSatSingleBatchSolver(OrtoolsCpSatSolver, WarmstartMixin):
    """CP-SAT Solver for the Single Batch-Processing Machine Scheduling Problem.

    Args:
        problem: The batch processing problem instance
        params_objective_function: Parameters for objective function
        modeling: Type of CP-SAT model (BINARY or SCHEDULING)
        symmetry_breaking: If True, applies symmetry breaking by ordering jobs
            by processing time and adding constraint x_jk <= x_kk for j <= k.
            From Trindade et al. (2018) "Modelling and symmetry breaking in
            scheduling problems on batch processing machines" (default: False)
    """

    problem: SingleBatchProcessingProblem

    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingBpm, default=ModelingBpm.BINARY
        )
    ]

    def __init__(
        self,
        problem,  # Assuming SingleBatchProcessingProblem
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.symmetry_breaking = False
        self.job_order = None
        self.job_order_inv = None

    def init_model(self, **args: Any) -> None:
        self.cp_model = CpModel()
        args = self.complete_with_default_hyperparameters(args)
        self.modeling = args["modeling"]
        self.symmetry_breaking = args.get("symmetry_breaking", False)
        self.max_batches = self.problem.nb_jobs

        if self.modeling == ModelingBpm.BINARY:
            self.init_model_binary(**args)
        elif self.modeling == ModelingBpm.SCHEDULING:
            self.init_model_scheduling(**args)

    def init_model_binary(self, **args: Any) -> None:
        if self.symmetry_breaking:
            self._init_model_binary_symmetry_breaking(**args)
        else:
            self._init_model_binary_naive(**args)

    def _init_model_binary_symmetry_breaking(self, **args: Any) -> None:
        """Symmetry breaking formulation from Trindade et al. (2018).

        Jobs are ordered by processing time and variables x_jk only exist for j <= k.
        """
        problem = self.problem
        nb_jobs = problem.nb_jobs

        # Sort jobs by processing time
        job_indices = list(range(nb_jobs))
        self.job_order_inv = sorted(
            job_indices, key=lambda j: (problem.jobs[j].processing_time, j)
        )
        self.job_order = [0] * nb_jobs
        for sorted_idx in range(nb_jobs):
            orig_idx = self.job_order_inv[sorted_idx]
            self.job_order[orig_idx] = sorted_idx

        # Variables: x[j, k] for j <= k only (using sorted indices)
        self.variables["x"] = {}
        for j_sorted in range(nb_jobs):
            j_orig = self.job_order_inv[j_sorted]
            for k_sorted in range(j_sorted, nb_jobs):
                self.variables["x"][j_orig, k_sorted] = self.cp_model.NewBoolVar(
                    f"x_{j_sorted}_{k_sorted}"
                )

        # Constraint (9): Each job assigned to exactly one batch
        for j_sorted in range(nb_jobs):
            j_orig = self.job_order_inv[j_sorted]
            self.cp_model.AddExactlyOne(
                [
                    self.variables["x"][j_orig, k_sorted]
                    for k_sorted in range(j_sorted, nb_jobs)
                ]
            )

        # Constraint (10): Capacity constraint using x_kk as batch indicator
        for k_sorted in range(nb_jobs):
            k_orig = self.job_order_inv[k_sorted]
            self.cp_model.Add(
                LinearExpr.WeightedSum(
                    [
                        self.variables["x"][self.job_order_inv[j_sorted], k_sorted]
                        for j_sorted in range(k_sorted + 1)
                    ],
                    [
                        problem.jobs[self.job_order_inv[j_sorted]].size
                        for j_sorted in range(k_sorted + 1)
                    ],
                )
                <= problem.capacity * self.variables["x"][k_orig, k_sorted]
            )

        # Constraint (11): Symmetry breaking x_jk <= x_kk
        for j_sorted in range(nb_jobs):
            j_orig = self.job_order_inv[j_sorted]
            for k_sorted in range(j_sorted + 1, nb_jobs):  # j < k only
                k_orig = self.job_order_inv[k_sorted]
                self.cp_model.AddImplication(
                    self.variables["x"][j_orig, k_sorted],
                    self.variables["x"][k_orig, k_sorted],
                )

        # Objective (8): Minimize sum of processing times of used batches
        makespan = self.cp_model.NewIntVar(
            0, sum(job.processing_time for job in problem.jobs), "makespan"
        )
        self.cp_model.Add(
            makespan
            == sum(
                problem.jobs[self.job_order_inv[k_sorted]].processing_time
                * self.variables["x"][self.job_order_inv[k_sorted], k_sorted]
                for k_sorted in range(nb_jobs)
            )
        )
        self.variables["makespan"] = makespan
        self.variables["y"] = None
        self.variables["batch_p"] = None
        self.cp_model.Minimize(makespan)

    def _init_model_binary_naive(self, **args: Any) -> None:
        """Standard formulation with separate batch and allocation variables."""
        problem = self.problem
        nb_jobs = problem.nb_jobs
        max_batches = self.max_batches

        # Variables
        self.variables["x"] = {}  # x[j, b] = 1 if job j is in batch b
        for j in range(nb_jobs):
            for b in range(max_batches):
                self.variables["x"][j, b] = self.cp_model.NewBoolVar(f"x_{j}_{b}")

        self.variables["y"] = {}  # y[b] = 1 if batch b is used
        for b in range(max_batches):
            self.variables["y"][b] = self.cp_model.NewBoolVar(f"y_{b}")

        max_p = max(job.processing_time for job in problem.jobs)
        self.variables["batch_p"] = {}  # batch_p[b] = processing time of batch b
        for b in range(max_batches):
            self.variables["batch_p"][b] = self.cp_model.NewIntVar(
                0, max_p, f"batch_p_{b}"
            )

        # 1. Assignment constraint
        for j in range(nb_jobs):
            self.cp_model.AddExactlyOne(
                [self.variables["x"][j, b] for b in range(max_batches)]
            )

        # 2. Capacity constraint (Linear)
        for b in range(max_batches):
            self.cp_model.Add(
                LinearExpr.WeightedSum(
                    [self.variables["x"][j, b] for j in range(nb_jobs)],
                    [problem.jobs[j].size for j in range(nb_jobs)],
                )
                <= problem.capacity * self.variables["y"][b]
            )

        # 3. Batch processing time definition
        for b in range(max_batches):
            for j in range(nb_jobs):
                self.cp_model.Add(
                    self.variables["batch_p"][b] >= problem.jobs[j].processing_time
                ).OnlyEnforceIf(self.variables["x"][j, b])
                self.cp_model.AddImplication(
                    self.variables["x"][j, b], self.variables["y"][b]
                )

        # 4. Symmetry Breaking
        for b in range(max_batches - 1):
            self.cp_model.AddImplication(
                self.variables["y"][b + 1], self.variables["y"][b]
            )

        # Objective
        makespan = self.cp_model.NewIntVar(
            0, sum(job.processing_time for job in problem.jobs), "makespan"
        )
        self.cp_model.Add(
            makespan == sum(self.variables["batch_p"][b] for b in range(max_batches))
        )
        self.variables["makespan"] = makespan
        self.cp_model.Minimize(makespan)

    def init_model_scheduling(self, **args: Any) -> None:
        problem = self.problem
        nb_jobs = problem.nb_jobs
        max_batches = self.max_batches

        # Variables
        self.variables["batch_idx"] = {}  # The batch index assigned to job j
        intervals = {}
        for j in range(nb_jobs):
            self.variables["batch_idx"][j] = self.cp_model.NewIntVar(
                0, max_batches - 1, f"batch_idx_{j}"
            )
            intervals[j] = self.cp_model.NewFixedSizeIntervalVar(
                start=self.variables["batch_idx"][j], size=1, name=f"interval_{j}"
            )

        # 1. Capacity constraint (Cumulative Scheduling)
        # We treat "batch index" as time. The cumulative sum at any "time" cannot exceed machine capacity.
        self.cp_model.AddCumulative(
            [intervals[j] for j in range(nb_jobs)],
            [problem.jobs[j].size for j in range(nb_jobs)],
            problem.capacity,
        )

        max_p = max(job.processing_time for job in problem.jobs)
        self.variables["batch_p"] = {}
        self.variables["is_in_batch"] = {}

        # 2. Extracting Batch Processing Time
        # We bridge the integer batch_idx[j] to boolean logic to determine the max duration per batch
        for b in range(max_batches):
            self.variables["batch_p"][b] = self.cp_model.NewIntVar(
                0, max_p, f"batch_p_{b}"
            )
            for j in range(nb_jobs):
                is_in = self.cp_model.NewBoolVar(f"is_in_{j}_{b}")
                self.variables["is_in_batch"][j, b] = is_in

                # Link boolean to integer batch index
                self.cp_model.Add(self.variables["batch_idx"][j] == b).OnlyEnforceIf(
                    is_in
                )
                self.cp_model.Add(self.variables["batch_idx"][j] != b).OnlyEnforceIf(
                    is_in.Not()
                )

                # Set batch duration
                self.cp_model.Add(
                    self.variables["batch_p"][b] >= problem.jobs[j].processing_time
                ).OnlyEnforceIf(is_in)

        # Objective
        makespan = self.cp_model.NewIntVar(
            0, sum(job.processing_time for job in problem.jobs), "makespan"
        )
        self.cp_model.Add(
            makespan == sum(self.variables["batch_p"][b] for b in range(max_batches))
        )
        self.variables["makespan"] = makespan
        self.cp_model.Minimize(makespan)

    def set_warm_start(self, solution: BatchProcessingSolution) -> None:
        self.cp_model.clear_hints()
        if self.modeling == ModelingBpm.BINARY:
            if self.symmetry_breaking:
                # Symmetry breaking formulation
                for j in range(self.problem.nb_jobs):
                    alloc = solution.job_to_batch[j]
                    if (j, alloc) in self.variables["x"]:
                        self.cp_model.add_hint(self.variables["x"][j, alloc], 1)
            else:
                # Naive formulation
                for j in range(self.problem.nb_jobs):
                    alloc = solution.job_to_batch[j]
                    self.cp_model.add_hint(self.variables["x"][j, alloc], 1)
                batch = set(solution.job_to_batch)
                for b in self.variables["y"]:
                    self.cp_model.add_hint(self.variables["y"][b], b in batch)
        elif self.modeling == ModelingBpm.SCHEDULING:
            for j in range(self.problem.nb_jobs):
                alloc = solution.job_to_batch[j]
                # Hint the batch index for job j
                self.cp_model.add_hint(self.variables["batch_idx"][j], alloc)
                # Hint the bridge booleans
                for b in range(self.max_batches):
                    self.cp_model.add_hint(
                        self.variables["is_in_batch"][j, b], 1 if alloc == b else 0
                    )

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        """Constructs a DO solution from the CP-SAT solver's internal state."""
        logger.info(
            f"Obj={cpsolvercb.ObjectiveValue()}, Bound={cpsolvercb.BestObjectiveBound()}"
        )
        problem = self.problem
        nb_jobs = problem.nb_jobs
        job_to_batch = [-1] * nb_jobs

        if self.modeling == ModelingBpm.BINARY:
            if self.symmetry_breaking:
                # Symmetry breaking: variables are (j_orig, k_sorted)
                for (j_orig, k_sorted), var in self.variables["x"].items():
                    if cpsolvercb.Value(var):
                        job_to_batch[j_orig] = k_sorted
            else:
                # Naive formulation
                for j in range(nb_jobs):
                    for b in range(self.max_batches):
                        if cpsolvercb.Value(self.variables["x"][j, b]):
                            job_to_batch[j] = b
                            break

        elif self.modeling == ModelingBpm.SCHEDULING:
            for j in range(nb_jobs):
                job_to_batch[j] = cpsolvercb.Value(self.variables["batch_idx"][j])

        return BatchProcessingSolution(problem=problem, job_to_batch=job_to_batch)
