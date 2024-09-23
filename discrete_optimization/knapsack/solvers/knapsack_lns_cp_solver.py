#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterable
from typing import Any, Optional

from minizinc import Instance
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.cp_tools import MinizincCPSolver
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    MznConstraintHandler,
    OrtoolsCPSatConstraintHandler,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN2
from discrete_optimization.knapsack.solvers.greedy_solvers import ResultStorage
from discrete_optimization.knapsack.solvers.knapsack_cpsat_solver import (
    CPSatKnapsackSolver,
)


class ConstraintHandlerKnapsack(MznConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: KnapsackModel, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MinizincCPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any
    ) -> Iterable[Any]:
        if not isinstance(solver, CPKnapsackMZN2):
            raise ValueError("cp_solver must a CPKnapsackMZN2 for this constraint.")
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a KnapsackSolution."
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


class OrtoolsCPSatConstraintHandlerKnapsack(OrtoolsCPSatConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: KnapsackModel, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: OrtoolsCPSatSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any
    ) -> Iterable[Constraint]:
        if not isinstance(solver, CPSatKnapsackSolver):
            raise ValueError(
                "cp_solver must a CPSatKnapsackSolver for this constraint."
            )
        lns_constraints = []
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a KnapsackSolution."
            )
        variables = solver.variables["taken"]
        for item in subpart_item:
            lns_constraints.append(
                solver.cp_model.Add(
                    variables[item] == current_solution.list_taken[item]
                )
            )
        return lns_constraints
