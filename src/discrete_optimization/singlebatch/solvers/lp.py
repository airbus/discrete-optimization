#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from ortools.math_opt.python import mathopt

try:
    import gurobipy
except ImportError:
    gurobipy = None

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.singlebatch.problem import (
    BatchProcessingSolution,
    SingleBatchProcessingProblem,
)

logger = logging.getLogger(__name__)


class BpmLpFormulation(Enum):
    """MILP formulation variants for batch processing machine scheduling."""

    NAIVE = 0  # Standard formulation with separate batch and allocation variables
    SYMMETRY_BREAKING = 1  # Trindade et al. (2018) formulation with reduced variables


class _BaseLpSingleBatchSolver(MilpSolver):
    """Base LP Solver for Single Batch-Processing Machine Scheduling Problem.

    Args:
        problem: The batch processing problem instance
        params_objective_function: Parameters for objective function
        formulation: Type of MILP formulation (NAIVE or SYMMETRY_BREAKING)
    """

    problem: SingleBatchProcessingProblem

    hyperparameters = [
        EnumHyperparameter(
            name="formulation", enum=BpmLpFormulation, default=BpmLpFormulation.NAIVE
        )
    ]

    def __init__(
        self,
        problem: SingleBatchProcessingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables = {}
        self.formulation = BpmLpFormulation.NAIVE
        self.job_order = None  # Maps original job index to sorted index
        self.job_order_inv = None  # Maps sorted index to original job index

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.formulation = kwargs["formulation"]
        self.model = self.create_empty_model("single_batch_processing")

        if self.formulation == BpmLpFormulation.SYMMETRY_BREAKING:
            self._init_model_symmetry_breaking(**kwargs)
        else:
            self._init_model_naive(**kwargs)

    def _init_model_symmetry_breaking(self, **kwargs: Any) -> None:
        """Symmetry breaking formulation from Trindade et al. (2018).

        Jobs are ordered by processing time and variables x_jk only exist for j <= k.
        x_kk = 1 indicates batch k is used.
        """
        n_jobs = self.problem.nb_jobs
        capacity = self.problem.capacity

        # Symmetry breaking formulation from Trindade et al. (2018)
        # Jobs ordered by processing time: p_0 <= p_1 <= ... <= p_{n-1}
        job_indices = list(range(n_jobs))
        self.job_order_inv = sorted(
            job_indices, key=lambda j: (self.problem.jobs[j].processing_time, j)
        )
        self.job_order = [0] * n_jobs
        for sorted_idx in range(n_jobs):
            orig_idx = self.job_order_inv[sorted_idx]
            self.job_order[orig_idx] = sorted_idx

        # Variables: x_jk for j <= k only
        # x_kk = 1 means batch k is used (job k assigned to batch k)
        variables_allocation = {}
        for j_sorted in range(n_jobs):
            j_orig = self.job_order_inv[j_sorted]
            for k_sorted in range(j_sorted, n_jobs):
                k_orig = self.job_order_inv[k_sorted]
                variables_allocation[(j_orig, k_sorted)] = self.add_binary_variable(
                    f"x_{j_sorted}_{k_sorted}"
                )

        # Constraint (9): Each job assigned to exactly one batch
        for j_sorted in range(n_jobs):
            j_orig = self.job_order_inv[j_sorted]
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        variables_allocation[(j_orig, k_sorted)]
                        for k_sorted in range(j_sorted, n_jobs)
                    ]
                )
                == 1,
                name=f"assign_job_{j_sorted}",
            )

        # Constraint (10): Capacity constraint using x_kk as batch indicator
        for k_sorted in range(n_jobs):
            k_orig = self.job_order_inv[k_sorted]
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        self.problem.jobs[self.job_order_inv[j_sorted]].size
                        * variables_allocation[(self.job_order_inv[j_sorted], k_sorted)]
                        for j_sorted in range(k_sorted + 1)
                    ]
                )
                <= capacity * variables_allocation[(k_orig, k_sorted)],
                name=f"capacity_{k_sorted}",
            )

        # Constraint (11): Symmetry breaking x_jk <= x_kk
        for j_sorted in range(n_jobs):
            j_orig = self.job_order_inv[j_sorted]
            for k_sorted in range(j_sorted, n_jobs):
                k_orig = self.job_order_inv[k_sorted]
                if (
                    j_sorted < k_sorted
                ):  # No need for j=k case (x_kk <= x_kk is trivial)
                    self.add_linear_constraint(
                        variables_allocation[(j_orig, k_sorted)]
                        <= variables_allocation[(k_orig, k_sorted)],
                        name=f"symm_break_{j_sorted}_{k_sorted}",
                    )

        # Objective (8): Minimize sum of processing times of used batches
        # Batch k is used iff x_kk = 1, and has processing time p_k
        self.set_model_objective(
            self.construct_linear_sum(
                [
                    self.problem.jobs[self.job_order_inv[k_sorted]].processing_time
                    * variables_allocation[(self.job_order_inv[k_sorted], k_sorted)]
                    for k_sorted in range(n_jobs)
                ]
            ),
            minimize=True,
        )

        self.variables["allocation"] = variables_allocation
        self.variables["used"] = None  # Not used in symmetry breaking formulation
        self.variables["batch_p"] = None  # Not used in symmetry breaking formulation

    def _init_model_naive(self, **kwargs: Any) -> None:
        """Standard formulation with separate batch and allocation variables."""
        n_jobs = self.problem.nb_jobs
        capacity = self.problem.capacity

        max_batches = n_jobs  # Worst case: 1 job per batch
        max_p = max(j.processing_time for j in self.problem.jobs)

        variables_allocation = {}
        used_batch = {}
        batch_p = {}

        # 1. Initialize Variables
        for b in range(max_batches):
            used_batch[b] = self.add_binary_variable(f"used_{b}")
            batch_p[b] = self.add_continuous_variable(
                lb=0.0, ub=max_p, name=f"batch_p_{b}"
            )

            # Symmetry breaking: Batches must be filled in order
            if b >= 1:
                self.add_linear_constraint(
                    used_batch[b] <= used_batch[b - 1], name=f"symm_used_{b}"
                )

        for j in range(n_jobs):
            for b in range(max_batches):
                variables_allocation[(j, b)] = self.add_binary_variable(
                    f"alloc_{j}_{b}"
                )

        # 2. Constraints
        for j in range(n_jobs):
            # Each job assigned to exactly one batch
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [variables_allocation[(j, b)] for b in range(max_batches)]
                )
                == 1,
                name=f"one_batch_for_job_{j}",
            )

        for b in range(max_batches):
            # Capacity constraint
            self.add_linear_constraint(
                self.construct_linear_sum(
                    [
                        self.problem.jobs[j].size * variables_allocation[(j, b)]
                        for j in range(n_jobs)
                    ]
                )
                <= capacity * used_batch[b],
                name=f"capacity_batch_{b}",
            )

            for j in range(n_jobs):
                # Link allocation to used_batch (redundant with capacity, but tightens LP)
                self.add_linear_constraint(
                    used_batch[b] >= variables_allocation[(j, b)],
                    name=f"link_used_{j}_{b}",
                )

                # Batch processing time is the max of the processing times of jobs in it
                self.add_linear_constraint(
                    batch_p[b]
                    >= self.problem.jobs[j].processing_time
                    * variables_allocation[(j, b)],
                    name=f"batch_duration_{j}_{b}",
                )

        self.variables["allocation"] = variables_allocation
        self.variables["used"] = used_batch
        self.variables["batch_p"] = batch_p

        # 3. Objective: Minimize Makespan (Sum of batch processing times)
        self.set_model_objective(
            self.construct_linear_sum([batch_p[b] for b in range(max_batches)]),
            minimize=True,
        )

    def convert_to_variable_values(
        self, solution: BatchProcessingSolution
    ) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values for warmstart."""
        hinted_variables = {var: 0.0 for var in self.variables["allocation"].values()}

        if self.formulation == BpmLpFormulation.SYMMETRY_BREAKING:
            # For symmetry breaking formulation, map solution through job ordering
            for j in range(self.problem.nb_jobs):
                b = solution.job_to_batch[j]
                # Find which batch index k_sorted this corresponds to
                j_sorted = self.job_order[j]
                # In symmetry breaking, we need to find a valid k_sorted >= j_sorted
                # The solution might not respect the ordering, so we remap batches
                if (j, b) in self.variables["allocation"]:
                    hinted_variables[self.variables["allocation"][(j, b)]] = 1.0
        else:
            # Naive formulation
            hinted_variables.update(
                {var: 0.0 for var in self.variables["used"].values()}
            )
            hinted_variables.update(
                {var: 0.0 for var in self.variables["batch_p"].values()}
            )

            batch_max_p = {}
            for j, b in enumerate(solution.job_to_batch):
                variable_decision_key = (j, b)
                hinted_variables[
                    self.variables["allocation"][variable_decision_key]
                ] = 1.0
                hinted_variables[self.variables["used"][b]] = 1.0

                # Compute real batch durations
                job_p = self.problem.jobs[j].processing_time
                batch_max_p[b] = max(batch_max_p.get(b, 0), job_p)

            for b, duration in batch_max_p.items():
                hinted_variables[self.variables["batch_p"][b]] = float(duration)

        return hinted_variables

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> BatchProcessingSolution:
        job_to_batch = [-1] * self.problem.nb_jobs

        if self.formulation == BpmLpFormulation.SYMMETRY_BREAKING:
            # In symmetry breaking: variables are (j_orig, k_sorted)
            # where x_jk = 1 means job j is in batch k
            for (j_orig, k_sorted), variable_decision_value in self.variables[
                "allocation"
            ].items():
                value = get_var_value_for_current_solution(variable_decision_value)
                if value >= 0.5:
                    job_to_batch[j_orig] = k_sorted
        else:
            # Naive formulation: variables are (j, b)
            for (j, b), variable_decision_value in self.variables["allocation"].items():
                value = get_var_value_for_current_solution(variable_decision_value)
                if value >= 0.5:
                    job_to_batch[j] = b

        return BatchProcessingSolution(self.problem, job_to_batch)


class MathOptSingleBatchSolver(_BaseLpSingleBatchSolver, OrtoolsMathOptMilpSolver):
    """MathOpt backend for the Single Batch LP Solver."""

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        return _BaseLpSingleBatchSolver.convert_to_variable_values(self, solution)


class GurobiSingleBatchSolver(_BaseLpSingleBatchSolver, GurobiMilpSolver):
    """Gurobi backend for the Single Batch LP Solver."""

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict["gurobipy.Var", float]:
        return _BaseLpSingleBatchSolver.convert_to_variable_values(self, solution)
