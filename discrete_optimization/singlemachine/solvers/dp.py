#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import didppy as dp

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, SolverDO
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


class DpWTSolver(DpSolver, WarmstartMixin):
    problem: WeightedTardinessProblem

    def __init__(
        self,
        problem: WeightedTardinessProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function=params_objective_function, **kwargs
        )
        self.transitions_names_to_info = {}
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        self.transitions_names_to_info = {}
        self.variables = {}
        model = dp.Model()
        tasks = model.add_object_type(number=self.problem.num_jobs)
        scheduled = model.add_set_var(object_type=tasks, target=[])
        all_jobs = model.create_set_const(
            object_type=tasks, value=list(range(self.problem.num_jobs))
        )
        processing_time = model.add_int_table(self.problem.processing_times)
        current_time = model.add_int_state_fun(processing_time[scheduled])
        for i in range(self.problem.num_jobs):
            tardiness = dp.max(
                0, current_time + processing_time[i] - self.problem.due_dates[i]
            )
            transition = dp.Transition(
                name=f"schedule_{i}",
                cost=self.problem.weights[i] * tardiness + dp.IntExpr.state_cost(),
                effects=[(scheduled, scheduled.add(i))],
                preconditions=[~scheduled.contains(i)],
            )
            model.add_transition(transition)
            self.transitions_names_to_info[f"schedule_{i}"] = {
                "task": i,
                "tr": transition,
            }
        model.add_base_case([scheduled == all_jobs])
        model.add_dual_bound(0)
        self.variables["current_time"] = current_time
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        schedule = [None for i in range(self.problem.num_jobs)]
        state = self.model.target_state
        for t in sol.transitions:
            current_time = self.variables["current_time"].eval(state, self.model)
            task = self.transitions_names_to_info[t.name]["task"]
            state = t.apply(state, self.model)
            new_time = self.variables["current_time"].eval(state, self.model)
            schedule[task] = (current_time, new_time)
        return WTSolution(problem=self.problem, schedule=schedule)

    def set_warm_start(self, solution: WTSolution) -> None:
        ordered_starts = sorted(
            range(len(solution.schedule)), key=lambda x: solution.schedule[x][0]
        )
        self.initial_solution = [
            self.transitions_names_to_info[f"schedule_{i}"]["tr"]
            for i in ordered_starts
        ]
