#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
import random
from typing import Any, Hashable, Iterable, Optional

from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.shop.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.shop.fjsp.solvers.cpsat import CpSatFjspSolver


class NeighborBuilderSubPart:
    """
    Cut the schedule in different subpart in the increasing order of the schedule.
    """

    def __init__(self, problem: FJobShopProblem, nb_cut_part: int = 10):
        self.problem = problem
        self.nb_cut_part = nb_cut_part
        self.current_sub_part = 0
        self.keys = self.problem.tasks_list
        self.set_keys = set(self.keys)

    def find_subtasks(
        self,
        current_solution: FJobShopSolution,
        subtasks: Optional[set[Hashable]] = None,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        nb_job_sub = math.ceil(self.problem.n_all_jobs / self.nb_cut_part)
        task_of_interest = sorted(
            self.keys, key=lambda x: current_solution.get_start_time(x)
        )
        task_of_interest = task_of_interest[
            self.current_sub_part * nb_job_sub : (self.current_sub_part + 1)
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
        self,
        solver: CpSatFjspSolver,
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
        sol: FJobShopSolution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        keys = random.sample(
            self.problem.tasks_list,
            k=int(self.fraction_segment_to_fix * len(self.problem.tasks_list)),
        )
        constraints = []
        for k in keys:
            st, end = sol.get_start_time(k), sol.get_end_time(k)
            constraints.append(
                solver.cp_model.Add(
                    solver.get_task_start_or_end_variable(k, StartOrEnd.START)
                    <= st + 20
                )
            )
            constraints.append(
                solver.cp_model.Add(
                    solver.get_task_start_or_end_variable(k, StartOrEnd.START)
                    >= st - 20
                )
            )
        solver.cp_model.Minimize(solver._makespan)
        solver.set_warm_start(sol)
        return constraints


class NeighFjspConstraintHandler(OrtoolsCpSatConstraintHandler):
    def __init__(
        self, problem: FJobShopProblem, neighbor_builder: NeighborBuilderSubPart
    ):
        self.problem = problem
        self.neighbor_builder = neighbor_builder

    def adding_constraint_from_results_store(
        self,
        solver: CpSatFjspSolver,
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
        sol, _ = result_storage[-1]
        sol: FJobShopSolution
        makespan = self.problem.evaluate(sol)["makespan"]
        keys_part, keys = self.neighbor_builder.find_subtasks(current_solution=sol)
        constraints = []
        for k in keys:
            st, end = sol.get_start_time(k), sol.get_end_time(k)
            constraints.append(
                solver.cp_model.Add(
                    solver.get_task_start_or_end_variable(k, StartOrEnd.START) <= st + 2
                )
            )
            constraints.append(
                solver.cp_model.Add(
                    solver.get_task_start_or_end_variable(k, StartOrEnd.START) >= st - 2
                )
            )
        for key in self.problem.tasks_list:
            constraints.append(
                solver.cp_model.Add(
                    solver.get_task_start_or_end_variable(key, StartOrEnd.END)
                    <= makespan
                )
            )
        solver.set_warm_start(sol)
        return constraints
