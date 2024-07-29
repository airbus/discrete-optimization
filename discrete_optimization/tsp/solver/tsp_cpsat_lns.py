#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Iterable

from ortools.sat.python.cp_model import Constraint
from discrete_optimization.tsp.solver.tsp_cpsat_solver import CpSatTspSolver
from discrete_optimization.tsp.tsp_model import TSPModel, SolutionTSP
from discrete_optimization.generic_tools.lns_cp import OrtoolsCPSatConstraintHandler
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import FloatHyperparameter
import random
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage


class ConstraintHandlerTSP(OrtoolsCPSatConstraintHandler):

    def __init__(self, problem: TSPModel, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(self, solver: CpSatTspSolver, result_storage: ResultStorage,
                                             **kwargs: Any) -> Iterable[Constraint]:
        sol, _ = result_storage.get_best_solution_fit()
        sol: SolutionTSP
        path = [sol.start_index] + sol.permutation + [sol.end_index, sol.start_index]
        constraints = []
        arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
        subpart = random.sample(arcs, k=int(self.fraction_segment_to_fix*len(arcs)))
        for s in subpart:
            constraints.append(solver.cp_model.Add(solver.variables["arc_literals"][s] == 1))
        return constraints


class ConstraintHandlerSubpathTSP(OrtoolsCPSatConstraintHandler):
    def __init__(self, problem: TSPModel, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(self, solver: CpSatTspSolver, result_storage: ResultStorage,
                                             **kwargs: Any) -> Iterable[Constraint]:
        sol, _ = result_storage.get_best_solution_fit()
        sol: SolutionTSP
        path = [sol.start_index] + sol.permutation + [sol.end_index, sol.start_index]
        constraints = []
        arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
        start_index = random.randint(0, len(arcs)-1)
        end_index = min(len(arcs), start_index+int(len(arcs)*(1-self.fraction_segment_to_fix)))
        for i in range(len(arcs)):
            if start_index <= i <= end_index:
                continue
            constraints.append(solver.cp_model.Add(solver.variables["arc_literals"][arcs[i]] == 1))
        return constraints