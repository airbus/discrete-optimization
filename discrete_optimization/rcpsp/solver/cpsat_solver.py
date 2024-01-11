#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Native ortools-cpsat implementation for multimode rcpsp with resource calendar.
#  Note : Model could most likely be improved with https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
import logging
from typing import Any, Dict, List, Optional

from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNKNOWN,
    CpModel,
    CpSolver,
    IntervalVar,
    VarArrayAndObjectiveSolutionPrinter,
    VarArraySolutionPrinter,
)

from discrete_optimization.generic_tools.cp_tools import ParametersCP, StatusSolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    from_solutions_to_result_storage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import create_fake_tasks

logger = logging.getLogger(__name__)


class CPSatRCPSPSolver(SolverDO):
    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.cp_model: Optional[CpModel] = None
        self.cp_solver: Optional[CpSolver] = None
        self.variables: Optional[Dict[str, Any]] = None
        self.status_solver: Optional[StatusSolver] = None

    def init_model(self):
        model = CpModel()
        starts_var = {}
        ends_var = {}
        is_present_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            starts_var[task] = model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"start_{task}"
            )
            ends_var[task] = model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"end_{task}"
            )
        interval_per_tasks = {}
        for task in self.problem.mode_details:
            interval_per_tasks[task] = set()
            for mode in self.problem.mode_details[task]:
                is_present_var[(task, mode)] = model.NewBoolVar(
                    f"is_present_{task,mode}"
                )
                interval_var[(task, mode)] = model.NewOptionalIntervalVar(
                    start=starts_var[task],
                    size=self.problem.mode_details[task][mode]["duration"],
                    end=ends_var[task],
                    is_present=is_present_var[(task, mode)],
                    name=f"interval_{task,mode}",
                )
                interval_per_tasks[task].add((task, mode))
        # Precedence constraints
        for task in self.problem.successors:
            for successor_task in self.problem.successors[task]:
                model.Add(starts_var[successor_task] >= ends_var[task])
        # 1 mode selected
        for task in interval_per_tasks:
            model.AddExactlyOne([is_present_var[k] for k in interval_per_tasks[task]])

        resources = self.problem.resources_list
        if self.problem.is_calendar:
            fake_task: List[Dict[str, int]] = create_fake_tasks(
                rcpsp_problem=self.problem
            )
        else:
            fake_task = []
        for resource in resources:
            if resource in self.problem.non_renewable_resources:
                task_modes_consuming = [
                    (
                        (task, mode),
                        self.problem.mode_details[task][mode].get(resource, 0),
                    )
                    for task in self.problem.tasks_list
                    for mode in self.problem.mode_details[task]
                    if self.problem.mode_details[task][mode].get(resource, 0) > 0
                ]
                model.Add(
                    sum([is_present_var[x[0]] * x[1] for x in task_modes_consuming])
                    <= self.problem.get_max_resource_capacity(resource)
                )

            else:
                task_modes_consuming = [
                    (
                        (task, mode),
                        self.problem.mode_details[task][mode].get(resource, 0),
                    )
                    for task in self.problem.tasks_list
                    for mode in self.problem.mode_details[task]
                    if self.problem.mode_details[task][mode].get(resource, 0) > 0
                ]
                fake_task_res = [
                    (
                        model.NewFixedSizeIntervalVar(
                            start=f["start"], size=f["duration"], name=f"res_"
                        ),
                        f.get(resource, 0),
                    )
                    for f in fake_task
                    if f.get(resource, 0) > 0
                ]
                capacity = self.problem.get_max_resource_capacity(resource)
                if capacity > 1:
                    model.AddCumulative(
                        [interval_var[x[0]] for x in task_modes_consuming]
                        + [x[0] for x in fake_task_res],
                        demands=[x[1] for x in task_modes_consuming]
                        + [x[1] for x in fake_task_res],
                        capacity=self.problem.get_max_resource_capacity(resource),
                    )
                if capacity == 1:
                    model.AddNoOverlap(
                        [interval_var[x[0]] for x in task_modes_consuming]
                        + [x[0] for x in fake_task_res]
                    )

        model.Minimize(starts_var[self.problem.sink_task])
        self.cp_model = model
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
        }

    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **kwargs: Any
    ) -> ResultStorage:
        if self.cp_model is None:
            self.init_model()
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        callback = VarArrayAndObjectiveSolutionPrinter(
            variables=list(self.variables["is_present"].values())
            + list(self.variables["start"].values())
        )
        status = solver.Solve(self.cp_model, callback)
        self.status_solver = cpstatus_to_dostatus(status_from_cpsat=status)
        logger.info(
            f"Solver finished, status={solver.StatusName(status)}, objective = {solver.ObjectiveValue()},"
            f"best obj bound = {solver.BestObjectiveBound()}"
        )
        return self.retrieve_solution(solver=solver)

    def retrieve_solution(self, solver: CpSolver):
        schedule = {}
        modes_dict = {}
        for task in self.variables["start"]:
            schedule[task] = {
                "start_time": solver.Value(self.variables["start"][task]),
                "end_time": solver.Value(self.variables["end"][task]),
            }
        for task, mode in self.variables["is_present"]:
            if solver.Value(self.variables["is_present"][task, mode]):
                modes_dict[task] = mode
        sol = RCPSPSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )
        return from_solutions_to_result_storage(
            [sol],
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )


def cpstatus_to_dostatus(status_from_cpsat) -> StatusSolver:
    """

    :param status_from_cpsat: either [UNKNOWN,INFEASIBLE,OPTIMAL,FEASIBLE] from ortools.cp api.
    :return: Status
    """
    if status_from_cpsat == UNKNOWN:
        return StatusSolver.UNKNOWN
    if status_from_cpsat == INFEASIBLE:
        return StatusSolver.UNSATISFIABLE
    if status_from_cpsat == OPTIMAL:
        return StatusSolver.OPTIMAL
    if status_from_cpsat == FEASIBLE:
        return StatusSolver.SATISFIED
