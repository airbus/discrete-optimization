#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    MultimodeOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution, Task
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.utils import create_fake_tasks

try:
    import optalcp as cp
except ImportError:
    cp = None


class OptalRcpspSolver(
    SchedulingOptalSolver[Task], MultimodeOptalSolver[Task], RcpspSolver, WarmstartMixin
):
    """Solver for RCPSP using the OptalCP TypeScript API (fallback if Python API is not available)"""

    problem: RcpspProblem

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables_dict = {}

    def create_fake_tasks(self) -> list[dict[str, int]]:
        """
        Create tasks representing the variable resource availability.
        :return:
        """
        if self.problem.is_calendar:
            fake_task: list[dict[str, int]] = create_fake_tasks(
                rcpsp_problem=self.problem
            )
        else:
            fake_task = []
        return fake_task

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        self.create_interval_vars()
        self.create_precedence_constraints()
        self.create_cumulative_constraints()
        self.cp_model.minimize(
            self.cp_model.end(self.variables_dict["intervals"][self.problem.sink_task])
        )

    def create_interval_vars(self):
        intervals = {}
        opt_intervals = {}
        for t in self.problem.tasks_list:
            modes = self.problem.mode_details[t]
            modes_keys = list(modes.keys())
            length = None
            if len(modes_keys) == 1:
                length = self.problem.mode_details[t][modes_keys[0]]["duration"]
            intervals[t] = self.cp_model.interval_var(
                start=(0, self.problem.horizon),
                end=(0, self.problem.horizon),
                length=length,
                optional=False,
                name=f"itv_{t}",
            )
            opt_intervals[t] = {}
            if len(modes_keys) == 1:
                opt_intervals[t][modes_keys[0]] = intervals[t]  # Useless
            else:
                for m in modes_keys:
                    length = self.problem.mode_details[t][m]["duration"]
                    opt_intervals[t][m] = self.cp_model.interval_var(
                        start=(0, self.problem.horizon),
                        end=(0, self.problem.horizon),
                        length=length,
                        optional=True,
                        name=f"itv_{t}_{m}",
                    )
                self.cp_model.alternative(
                    intervals[t], [opt_intervals[t][m] for m in opt_intervals[t]]
                )
        self.variables_dict["intervals"] = intervals
        self.variables_dict["opt_intervals"] = opt_intervals

    def create_precedence_constraints(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.end_before_start(
                    self.variables_dict["intervals"][t],
                    self.variables_dict["intervals"][succ],
                )

    def create_cumulative_constraints(self):
        calendar_tasks = self.create_fake_tasks()
        for res in self.problem.resources_list:
            capa = self.problem.get_max_resource_capacity(res)
            if res not in self.problem.non_renewable_resources:
                # The calendar virtual intervals.
                list_pulse = [
                    self.cp_model.pulse(
                        interval=self.cp_model.interval_var(
                            start=x["start"],
                            end=x["start"] + x["duration"],
                            optional=False,
                        ),
                        height=x[res],
                    )
                    for x in calendar_tasks
                    if x.get(res, 0) > 0
                ]
                for t in self.variables_dict["opt_intervals"]:
                    for m in self.variables_dict["opt_intervals"][t]:
                        conso = self.problem.mode_details[t][m].get(res, 0)
                        if conso > 0:
                            list_pulse.append(
                                self.cp_model.pulse(
                                    interval=self.variables_dict["opt_intervals"][t][m],
                                    height=conso,
                                )
                            )
                self.cp_model.enforce(self.cp_model.sum(list_pulse) <= capa)
            else:
                list_conso = []
                for t in self.variables_dict["opt_intervals"]:
                    for m in self.variables_dict["opt_intervals"][t]:
                        conso = self.problem.mode_details[t][m].get(res, 0)
                        if conso > 0:
                            list_conso.append(
                                self.cp_model.presence(
                                    self.variables_dict["opt_intervals"][t][m]
                                )
                                * conso
                            )
                self.cp_model.enforce(self.cp_model.sum(list_conso) <= capa)

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables_dict["intervals"][task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> cp.BoolExpr:
        return self.cp_model.presence(self.variables_dict["opt_intervals"][task][mode])

    def set_warm_start(self, solution: RcpspSolution) -> None:
        solution_optal = cp.Solution()
        for task in self.problem.tasks_list:
            st = solution.get_start_time(task)
            end = solution.get_end_time(task)
            mode = solution.get_mode(task)
            solution_optal.set_value(self.get_task_interval_variable(task), st, end)
            modes = self.problem.get_task_modes(task)
            if len(modes) > 0:
                solution_optal.set_value(
                    self.variables_dict["opt_intervals"][task][mode], st, end
                )
        makespan = solution.get_max_end_time()
        solution_optal.set_objective(makespan)
        self.warm_start_solution = solution_optal
        self.use_warm_start = True

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        schedule = {}
        modes_dict = {}
        if result.solution is not None:
            for t in self.variables_dict["intervals"]:
                start = result.solution.get_start(self.variables_dict["intervals"][t])
                end = result.solution.get_end(self.variables_dict["intervals"][t])
                schedule[t] = {"start_time": start, "end_time": end}
                for m in self.variables_dict["opt_intervals"][t]:
                    if result.solution.is_present(
                        self.variables_dict["opt_intervals"][t][m]
                    ):
                        modes_dict[t] = m
            return RcpspSolution(
                problem=self.problem,
                rcpsp_schedule=schedule,
                rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
            )
        return None
