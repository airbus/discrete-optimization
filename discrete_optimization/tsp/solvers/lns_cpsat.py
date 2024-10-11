#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from collections.abc import Iterable
from typing import Any

from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver


class TspConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(self, problem: TspProblem, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(
        self, solver: CpSatTspSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        sol, _ = result_storage.get_best_solution_fit()
        sol: TspSolution
        if sol.end_index != sol.start_index:
            path = (
                [sol.start_index] + sol.permutation + [sol.end_index, sol.start_index]
            )
        else:
            path = [sol.start_index] + sol.permutation + [sol.end_index]
        constraints = []
        arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
        subpart = random.sample(arcs, k=int(self.fraction_segment_to_fix * len(arcs)))
        for s in subpart:
            constraints.append(
                solver.cp_model.Add(solver.variables["arc_literals"][s] == 1)
            )
        return constraints


class SubpathTspConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(self, problem: TspProblem, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(
        self, solver: CpSatTspSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        sol, _ = result_storage.get_best_solution_fit()
        sol: TspSolution
        if sol.end_index != sol.start_index:
            path = (
                [sol.start_index] + sol.permutation + [sol.end_index, sol.start_index]
            )
        else:
            path = [sol.start_index] + sol.permutation + [sol.end_index]
        constraints = []
        arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
        start_index = random.randint(0, len(arcs) - 1)
        end_index = min(
            len(arcs), start_index + int(len(arcs) * (1 - self.fraction_segment_to_fix))
        )
        for i in range(len(arcs)):
            if start_index <= i <= end_index:
                continue
            constraints.append(
                solver.cp_model.Add(solver.variables["arc_literals"][arcs[i]] == 1)
            )
        return constraints
