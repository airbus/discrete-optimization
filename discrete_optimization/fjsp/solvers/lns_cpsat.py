#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
import random
from typing import Any, Hashable, Iterable, Optional

from ortools.sat.python.cp_model import Constraint

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class NeighborBuilderSubPart:
    """
    Cut the schedule in different subpart in the increasing order of the schedule.
    """

    def __init__(self, problem: FJobShopProblem, nb_cut_part: int = 10):
        self.problem = problem
        self.nb_cut_part = nb_cut_part
        self.current_sub_part = 0
        self.keys = [
            (i, j)
            for i in range(self.problem.n_jobs)
            for j in range(len(self.problem.list_jobs[i].sub_jobs))
        ]
        self.set_keys = set(self.keys)

    def find_subtasks(
        self,
        current_solution: FJobShopSolution,
        subtasks: Optional[set[Hashable]] = None,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        nb_job_sub = math.ceil(self.problem.n_all_jobs / self.nb_cut_part)
        task_of_interest = sorted(
            self.keys, key=lambda x: current_solution.schedule[x[0]][x[1]][0]
        )
        task_of_interest = task_of_interest[
            self.current_sub_part
            * nb_job_sub : (self.current_sub_part + 1)
            * nb_job_sub
        ]
        if subtasks is None:
            subtasks = task_of_interest
        else:
            subtasks.update(task_of_interest)
        self.current_sub_part = (self.current_sub_part + 1) % self.nb_cut_part
        return subtasks, self.set_keys.difference(subtasks)


class FjspConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(self, problem: FJobShopProblem, fraction_segment_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_segment_to_fix = fraction_segment_to_fix

    def adding_constraint_from_results_store(
        self, solver: CpSatFjspSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        sol, _ = result_storage.get_best_solution_fit()
        sol: FJobShopSolution
        keys = random.sample(
            list(solver.variables["starts"]),
            k=int(self.fraction_segment_to_fix * len(solver.variables["starts"])),
        )
        constraints = []
        for k in keys:
            st, end, machine = sol.schedule[k[0]][k[1]]
            constraints.append(
                solver.cp_model.Add(solver.variables["starts"][k] <= st + 20)
            )
            constraints.append(
                solver.cp_model.Add(solver.variables["starts"][k] >= st - 20)
            )
            # constraints.append(solver.cp_model.Add(solver.variables["ends"][k] <= end+20))
            # constraints.append(solver.cp_model.Add(solver.variables["ends"][k] >= end-20))
        solver.cp_model.Minimize(solver.variables["makespan"])
        solver.set_warm_start(sol)
        return constraints


class NeighFjspConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(
        self, problem: FJobShopProblem, neighbor_builder: NeighborBuilderSubPart
    ):
        self.problem = problem
        self.neighbor_builder = neighbor_builder

    def adding_constraint_from_results_store(
        self, solver: CpSatFjspSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        sol, _ = result_storage[-1]
        sol: FJobShopSolution
        makespan = self.problem.evaluate(sol)["makespan"]
        keys_part, keys = self.neighbor_builder.find_subtasks(current_solution=sol)
        constraints = []
        for k in keys:
            st, end, machine = sol.schedule[k[0]][k[1]]
            constraints.append(
                solver.cp_model.Add(solver.variables["starts"][k] <= st + 2)
            )
            constraints.append(
                solver.cp_model.Add(solver.variables["starts"][k] >= st - 2)
            )
            # constraints.append(solver.cp_model.Add(solver.variables["ends"][k] <= end+2))
            # constraints.append(solver.cp_model.Add(solver.variables["ends"][k] >= end-2))
        for key in solver.variables["ends"]:
            constraints.append(
                solver.cp_model.Add(solver.variables["ends"][key] <= makespan)
            )
        # m = solver.cp_model.NewIntVar(lb=0, ub=makespan, name="")
        # solver.cp_model.AddMaxEquality(m, [solver.variables["ends"][k] for k in keys_part])
        # solver.cp_model.Minimize(m) #+sum([solver.variables["ends"][k] for k in keys_part]))
        solver.set_warm_start(sol)
        return constraints
