#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from typing import Any, Callable, List

from ortools.sat.python import cp_model

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver

logger = logging.getLogger(__name__)


class CpsatParetoSolver(SolverDO):
    """
    Finds the Pareto front for N objectives using Iterative Blocking with Lexicographic Tightening.

    Algorithm:
    1. Search for a feasible solution (optimizing primary objective).
    2. 'Tighten' the solution to ensure Pareto optimality (Lexicographic minimization chain).
    3. Record the solution.
    4. Add a 'Blocking Clause' to the model: Next solution must be strictly better
       in at least one objective.
    5. Repeat until Infeasible.
    """

    def __init__(
        self,
        problem: Problem,
        solver: OrtoolsCpSatSolver,
        objective_names: List[str],
        dict_function: dict[str, Callable[[Solution], float]] = None,
        delta_abs_improvement: list[int] = None,
        delta_ref_improvement: list[float] = None,
    ):
        """
        Args:
            solver: The base CP-SAT solver instance.
            objective_names: list of objectives as string, to consider in the pareto
            dict_function: a map of function to compute objective kpi on a solution
            delta_abs_improvement: list of absolute improvement for each objective to add in the blocking clauses
            delta_ref_improvement: list of relative improvement for each objective to add in the blocking clauses
        """
        super().__init__(problem)
        self.problem = problem
        self.solver = solver
        self.objective_names = objective_names
        if dict_function is None:
            dict_function = {
                key: lambda sol: self.problem.evaluate(sol)[key]
                for key in self.objective_names
            }
        self.dict_function = dict_function
        if delta_abs_improvement is None:
            delta_abs_improvement = [1 for _ in range(len(objective_names))]
        if delta_ref_improvement is None:
            delta_ref_improvement = [0.01 for _ in range(len(objective_names))]
        self.delta_abs_improvement = delta_abs_improvement
        self.delta_ref_improvement = delta_ref_improvement

    def solve(
        self,
        obj_vars: list[cp_model.LinearExpr],
        callbacks: list[Callback] = None,
        time_limit: int = None,
        subsolver_kwargs: dict = None,
        **kwargs,
    ):
        if subsolver_kwargs is None:
            subsolver_kwargs = {}
        # 1. Initialize Model
        # self.solver.init_model()
        pareto_front = []
        objectives = obj_vars
        t = time.perf_counter()
        while True and time.perf_counter() - t < time_limit:
            logger.info(f"--- Iteration: Searching for non-dominated solutions ---")
            # --- Step 1: Find a point on the pareto frontier with first objective ---
            # We minimize the first objective to find *some* point on the frontier.
            # (Constraints from previous steps ensure we don't find known points).
            self.solver.set_lexico_objective(self.objective_names[0])
            try:
                self.solver.set_warm_start_from_previous_run()
            except RuntimeError:
                pass
            res = self.solver.solve(**subsolver_kwargs)
            status = self.solver.status_solver
            if status in {StatusSolver.UNSATISFIABLE, StatusSolver.UNKNOWN}:
                logger.info("  -> Infeasible. Pareto Front Search Complete.")
                break
            # Get candidate solution
            sol_candidate = res.get_best_solution()
            print(status, sol_candidate)
            vals_candidate = [
                self.dict_function[obj](sol_candidate) for obj in self.objective_names
            ]
            logger.info(f"  -> Candidate found (pre-tightening): {vals_candidate}")
            # --- Step 2: Lexicographic Tightening ---
            # Ensure the point is strictly Pareto optimal.
            # Chain: Fix f1, Min f2 -> Fix f1,f2, Min f3 -> ...
            tightening_constraints = []
            current_vals = list(vals_candidate)  # We update these as we tighten
            sol_lex = None
            fit_lex = None
            for i in range(len(objectives)):
                # For the objective we are about to optimize (obj[i]), we don't fix it yet.
                # For all PREVIOUS objectives (0 to i-1), we fix them to their best found values.
                if i > 0:
                    prev_var = objectives[i - 1]
                    prev_val = int(current_vals[i - 1])
                    # Add constraint: obj[i-1] <= val
                    c = self.solver.add_bound_constraint(
                        prev_var, SignEnum.LEQ, prev_val
                    )
                    tightening_constraints.extend(c)
                # Now minimize current objective i
                self.solver.minimize_variable(objectives[i])
                # Solve
                self.solver.set_warm_start_from_previous_run()
                res_lex = self.solver.solve(**subsolver_kwargs)
                status = self.solver.status_solver
                if status in {StatusSolver.UNSATISFIABLE, StatusSolver.UNKNOWN}:
                    logger.warning(
                        "  -> Tightening failed (Infeasible). Keeping previous candidate."
                    )
                    break
                # Update current best values based on this tightening step
                # Again, re-evaluate via problem
                sol_lex, fit_lex = res_lex.get_best_solution_fit()
                current_vals = [
                    self.dict_function[obj](sol_lex) for obj in self.objective_names
                ]
            # Capture Final Tightened Solution
            # Usually the last solution in the chain is the Tightened one
            logger.info(f"  -> Tightened Result: {current_vals}")
            pareto_front.append((sol_lex, fit_lex))
            if tightening_constraints:
                self.solver.remove_constraints(tightening_constraints)
            # Block regions dominated by or equal to the found values
            self._add_blocking_constraint(objectives, current_vals)
        return pareto_front

    def _add_blocking_constraint(self, objectives: List[Any], values: List[float]):
        """
        Adds a permanent constraint: OR(obj[i] <= val[i] - 1) for all i.
        This blocks any solution that is 'worse or equal' in all dimensions.
        """
        literals = []
        for i, obj_var in enumerate(objectives):
            val = int(values[i])
            # New boolean for this implication
            lit = self.solver.cp_model.NewBoolVar(f"block_{len(literals)}_{val}")
            # Enforce: lit => obj_var <= val - 1
            # (Strict improvement in this objective)
            self.solver.cp_model.Add(
                obj_var
                <= val
                - max(
                    self.delta_abs_improvement[i],
                    int(self.delta_ref_improvement[i] * val),
                )
            ).OnlyEnforceIf(lit)
            literals.append(lit)
        # Enforce: At least one of these strict improvements must happen
        self.solver.cp_model.AddBoolOr(literals)
