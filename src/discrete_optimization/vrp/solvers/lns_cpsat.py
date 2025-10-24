#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random
from collections.abc import Iterable
from typing import Any

from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrp.problem import VrpProblem, VrpSolution
from discrete_optimization.vrp.solvers.cpsat import CpSatVrpSolver


class VrpConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(self, problem: VrpProblem, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: CpSatVrpSolver,
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
        sol: VrpSolution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        constraints = []
        for v in range(len(sol.list_paths)):
            path = (
                [sol.list_start_index[v]] + sol.list_paths[v] + [sol.list_end_index[v]]
            )
            arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
            subpart = random.sample(
                arcs, k=int(self.fraction_segment_to_fix * len(arcs))
            )
            for s in subpart:
                constraints.append(
                    solver.cp_model.Add(
                        solver.variables["arc_literals_per_vehicles"][v][s] == 1
                    )
                )
        return constraints


class SubpathVrpConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(self, problem: VrpProblem, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: CpSatVrpSolver,
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
        sol: VrpSolution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        constraints = []
        for v in range(len(sol.list_paths)):
            path = (
                [sol.list_start_index[v]] + sol.list_paths[v] + [sol.list_end_index[v]]
            )
            arcs = [(x, y) for x, y in zip(path[:-1], path[1:])]
            if len(arcs) >= 1:
                start_index = random.randint(0, len(arcs) - 1)
                end_index = min(
                    len(arcs),
                    start_index + int(len(arcs) * (1 - self.fraction_segment_to_fix)),
                )
                for i in range(len(arcs)):
                    if start_index <= i <= end_index:
                        continue
                    constraints.append(
                        solver.cp_model.Add(
                            solver.variables["arc_literals_per_vehicles"][v][arcs[i]]
                            == 1
                        )
                    )
        return constraints
