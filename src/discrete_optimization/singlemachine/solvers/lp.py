#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Callable, Dict

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    InequalitySense,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)

try:
    import gurobipy
except:
    pass


logger = logging.getLogger(__name__)


class _BaseLpSingleMachineSolver(MilpSolver):
    """
    LP solver for the Single Machine Weighted Tardiness Problem.

    This solver uses a relative ordering formulation with 'big-M' constraints
    to ensure non-overlapping schedules.
    """

    def __init__(self, problem: WeightedTardinessProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)
        self.problem: WeightedTardinessProblem = problem
        self.variables: Dict[str, Any] = {}

    def init_model(self, **kwargs: Any) -> None:
        """
        Builds the LP model from the WeightedTardinessProblem instance.
        It defines variables, constraints, and the objective function.
        """
        # Create a new empty model
        self.model = self.create_empty_model()

        # Problem Data
        num_jobs = self.problem.num_jobs
        p = self.problem.processing_times
        w = self.problem.weights
        d = self.problem.due_dates
        r = self.problem.release_dates

        # --- Define Variables ---
        # x_ij = 1 if job i is before job j
        x = {
            (i, j): self.add_binary_variable(name=f"x_{i},{j}")
            for i in range(num_jobs)
            for j in range(num_jobs)
            if i != j
        }
        # C_i = completion time of job i
        c = {
            i: self.add_continuous_variable(lb=r[i] + p[i], name=f"C_{i}")
            for i in range(num_jobs)
        }
        # T_i = tardiness of job i
        t = {
            i: self.add_continuous_variable(lb=0, name=f"T_{i}")
            for i in range(num_jobs)
        }
        self.variables = {"x": x, "c": c, "t": t}

        big_m = sum(p) + max(r)

        # Ordering Constraints
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                self.add_linear_constraint(x[(i, j)] + x[(j, i)] == 1)

        # Completion time constraints (Big-M for non-overlap)
        for i in range(num_jobs):
            for j in range(num_jobs):
                if i == j:
                    continue
                self.add_linear_constraint_with_indicator(
                    x[(i, j)],
                    1,
                    c[j],
                    sense=InequalitySense.GREATER_OR_EQUAL,
                    rhs=c[i] + p[j],
                    penalty_coeff=big_m,
                )

        # Tardiness definition: T_i >= C_i - d_i
        for i in range(num_jobs):
            self.add_linear_constraint(t[i] >= c[i] - d[i])

        # 5. Transitivity constraints: x_ij + x_jk + x_ki <= 2
        for i in range(num_jobs):
            for j in range(num_jobs):
                if i == j:
                    continue
                for k in range(num_jobs):
                    if k in {i, j}:
                        continue
                    self.add_linear_constraint(x[(i, j)] + x[(j, k)] + x[(k, i)] <= 2)

        # --- Define Objective Function ---
        objective = self.construct_linear_sum([w[i] * t[i] for i in range(num_jobs)])
        self.set_model_objective(objective, minimize=True)
        logger.info("LP model initialized.")

    def retrieve_current_solution(
        self, get_var_value_for_current_solution: Callable[[Any], float], **kwargs
    ) -> Solution:
        """
        Retrieves a WTSolution from the LP solver's results.

        Args:
            get_var_value_for_current_solution: A function provided by the
                MilpSolver framework to query the value of a model variable.

        Returns:
            A WTSolution object representing the schedule.
            :param **kwargs:
        """
        schedule = [() for _ in range(self.problem.num_jobs)]
        for i in range(self.problem.num_jobs):
            completion_time = get_var_value_for_current_solution(self.variables["c"][i])
            start_time = completion_time - self.problem.processing_times[i]
            schedule[i] = (int(round(start_time)), int(round(completion_time)))

        return WTSolution(problem=self.problem, schedule=schedule)


class MathOptSingleMachineSolver(_BaseLpSingleMachineSolver, OrtoolsMathOptMilpSolver):
    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        pass


class GurobiSingleMachineSolver(_BaseLpSingleMachineSolver, GurobiMilpSolver):
    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict["gurobipy.Var", float]:
        pass
