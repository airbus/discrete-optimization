#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterable
from typing import Any, Optional

from minizinc import Instance
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.cp_tools import MinizincCpSolver
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    MznConstraintHandler,
    OrtoolsCpSatConstraintHandler,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers.cp_mzn import Cp2KnapsackSolver
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver
from discrete_optimization.knapsack.solvers.greedy import ResultStorage


class KnapsackMznConstraintHandler(MznConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: KnapsackProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MinizincCpSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        child_instance: Instance,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            child_instance: minizinc instance where to include the constraints
            **kwargs:

        Returns:
            empty list, not used

        """
        if not isinstance(solver, Cp2KnapsackSolver):
            raise ValueError("cp_solver must a CPKnapsackMZN2 for this constraint.")
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a KnapsackSolution."
            )
        list_strings = []
        for item in subpart_item:
            list_strings += [
                "constraint taken["
                + str(item + 1)
                + "] == "
                + str(current_solution.list_taken[item])
                + ";\n"
            ]
            child_instance.add_string(list_strings[-1])
        return list_strings


class OrtoolsCpSatKnapsackConstraintHandler(OrtoolsCpSatConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: KnapsackProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: OrtoolsCpSatSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            **kwargs:

        Returns:
            list of added constraints

        """
        if not isinstance(solver, CpSatKnapsackSolver):
            raise ValueError(
                "cp_solver must a CpSatKnapsackSolver for this constraint."
            )
        lns_constraints = []
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        if current_solution is None:
            raise ValueError("result_storage.get_best_solution() should not be None.")
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a KnapsackSolution."
            )
        variables = solver.variables["taken"]
        for item in subpart_item:
            lns_constraints.append(
                solver.cp_model.Add(
                    variables[item] == current_solution.list_taken[item]
                )
            )
        return lns_constraints
