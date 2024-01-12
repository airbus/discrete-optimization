#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Native ortools-cpsat implementation for multimode rcpsp with resource calendar.
#  Note : Model could most likely be improved with
#  https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
import logging
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNKNOWN,
    CpModel,
    CpSolver,
    VarArrayAndObjectiveSolutionPrinter,
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
from discrete_optimization.rcpsp.rcpsp_utils import (
    create_fake_tasks,
    get_end_bounds_from_additional_constraint,
    get_start_bounds_from_additional_constraint,
)

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

    def init_model(self, **kwargs):
        modes_allowed_assignment: Dict[
            Tuple[Hashable, Hashable], List[Tuple[Hashable, Hashable]]
        ] = kwargs.get("modes_allowed_assignment", {})
        task_mode_score: Dict[Tuple[Hashable, Hashable], int] = kwargs.get(
            "task_mode_score", {}
        )
        model = CpModel()
        starts_var = {}
        ends_var = {}
        is_present_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            lbs, ubs = get_start_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            lbe, ube = get_end_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            starts_var[task] = model.NewIntVar(lb=lbs, ub=ubs, name=f"start_{task}")
            ends_var[task] = model.NewIntVar(lb=lbe, ub=ube, name=f"end_{task}")
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
        score_task = {}

        for task in set([x[0] for x in task_mode_score]):
            min_score = min(task_mode_score[x] for x in interval_per_tasks[task])
            max_score = max(task_mode_score[x] for x in interval_per_tasks[task])
            score_task[task] = model.NewIntVar(
                lb=min_score, ub=max_score, name=f"score_{task}"
            )
            model.Add(
                score_task[task]
                == sum(
                    [
                        task_mode_score[x] * is_present_var[x]
                        for x in interval_per_tasks[task]
                    ]
                )
            )
        for task1, task2 in modes_allowed_assignment:
            # way 1
            model.Add(score_task[task1] == score_task[task2])

            # way 2
            for mode1, mode2 in modes_allowed_assignment[(task1, task2)]:
                model.AddAllowedAssignments(
                    [is_present_var[(task1, mode1)], is_present_var[(task2, mode2)]],
                    [(1, 1), (0, 0)],
                )

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
        callback = VarArrayAndObjectiveSolutionPrinter(variables=[])
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


class CPSatRCPSPSolverResource(SolverDO):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we don't sum up the resource at a given time, and it suits/makes sense mostly
    for disjunctive resource (machines)
    """

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

    def init_model(self, **kwargs):
        modes_allowed_assignment: Dict[
            Tuple[Hashable, Hashable], List[Tuple[Hashable, Hashable]]
        ] = kwargs.get("modes_allowed_assignment", {})
        task_mode_score: Dict[Tuple[Hashable, Hashable], int] = kwargs.get(
            "task_mode_score", {}
        )
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)
        model = CpModel()
        starts_var = {}
        ends_var = {}
        is_present_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            lbs, ubs = get_start_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            lbe, ube = get_end_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            starts_var[task] = model.NewIntVar(lb=lbs, ub=ubs, name=f"start_{task}")
            ends_var[task] = model.NewIntVar(lb=lbe, ub=ube, name=f"end_{task}")
        interval_per_tasks = {}
        for task in self.problem.mode_details:
            interval_per_tasks[task] = set()
            for mode in self.problem.mode_details[task]:
                is_present_var[(task, mode)] = model.NewBoolVar(
                    f"is_present_{task,mode}"
                )
                interval_var[(task, mode)] = model.NewOptionalIntervalVar(
                    start=starts_var[task],
                    size=int(self.problem.mode_details[task][mode]["duration"]),
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
        score_task = {}
        for task in set([x[0] for x in task_mode_score]):
            min_score = min(task_mode_score[x] for x in interval_per_tasks[task])
            max_score = max(task_mode_score[x] for x in interval_per_tasks[task])
            score_task[task] = model.NewIntVar(
                lb=int(min_score), ub=int(max_score), name=f"score_{task}"
            )
            model.Add(
                score_task[task]
                == sum(
                    [
                        task_mode_score[x] * is_present_var[x]
                        for x in interval_per_tasks[task]
                    ]
                )
            )
        for task1, task2 in modes_allowed_assignment:
            # way 1
            model.Add(score_task[task1] == score_task[task2])
            # way 2
            for mode1, mode2 in modes_allowed_assignment[(task1, task2)]:
                model.AddAllowedAssignments(
                    [is_present_var[(task1, mode1)], is_present_var[(task2, mode2)]],
                    [(1, 1), (0, 0)],
                )
        is_used_resource = {
            res: model.NewBoolVar(f"used_{res}") for res in self.problem.resources_list
        }
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
                        int(self.problem.mode_details[task][mode].get(resource, 0)),
                    )
                    for task in self.problem.tasks_list
                    for mode in self.problem.mode_details[task]
                    if self.problem.mode_details[task][mode].get(resource, 0) > 0
                ]
                model.Add(
                    sum([is_present_var[x[0]] * x[1] for x in task_modes_consuming])
                    <= int(self.problem.get_max_resource_capacity(resource))
                )
                model.AddMaxEquality(
                    is_used_resource[resource],
                    [is_present_var[x[0]] for x in task_modes_consuming],
                )
            else:
                task_modes_consuming = [
                    (
                        (task, mode),
                        int(self.problem.mode_details[task][mode].get(resource, 0)),
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
                        int(f.get(resource, 0)),
                    )
                    for f in fake_task
                    if f.get(resource, 0) > 0
                ]
                capacity = int(self.problem.get_max_resource_capacity(resource))
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
                model.AddMaxEquality(
                    is_used_resource[resource],
                    [is_present_var[x[0]] for x in task_modes_consuming],
                )
        model.Minimize(
            weight_on_used_resource
            * sum([is_used_resource[x] for x in is_used_resource])
            + weight_on_makespan * starts_var[self.problem.sink_task]
        )
        self.cp_model = model
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "is_used_resource": is_used_resource,
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
            variables=list(self.variables["is_used_resource"].values())
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


class CPSatRCPSPSolverCumulativeResource(SolverDO):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we sum up the resource for each given time to do the resource optimisation.
    """

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

    def init_model(self, **kwargs):
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)
        modes_allowed_assignment: Dict[
            Tuple[Hashable, Hashable], List[Tuple[Hashable, Hashable]]
        ] = kwargs.get("modes_allowed_assignment", {})
        task_mode_score: Dict[Tuple[Hashable, Hashable], int] = kwargs.get(
            "task_mode_score", {}
        )
        use_overlap_for_disjunctive_resource = kwargs.get(
            "use_overlap_for_disjunctive_resource", True
        )
        model = CpModel()
        starts_var = {}
        ends_var = {}
        is_present_var = {}
        interval_var = {}
        resource_capacity_var = {}
        for resource in self.problem.resources_list:
            resource_capacity_var[resource] = model.NewIntVar(
                lb=0,
                ub=self.problem.get_max_resource_capacity(resource),
                name=f"res_{resource}",
            )
        for task in self.problem.tasks_list:
            lbs, ubs = get_start_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            lbe, ube = get_end_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            starts_var[task] = model.NewIntVar(lb=lbs, ub=ubs, name=f"start_{task}")
            ends_var[task] = model.NewIntVar(lb=lbe, ub=ube, name=f"end_{task}")
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
        score_task = {}
        for task in set([x[0] for x in task_mode_score]):
            min_score = min(task_mode_score[x] for x in interval_per_tasks[task])
            max_score = max(task_mode_score[x] for x in interval_per_tasks[task])
            score_task[task] = model.NewIntVar(
                lb=min_score, ub=max_score, name=f"score_{task}"
            )
            model.Add(
                score_task[task]
                == sum(
                    [
                        task_mode_score[x] * is_present_var[x]
                        for x in interval_per_tasks[task]
                    ]
                )
            )
        for task1, task2 in modes_allowed_assignment:
            # way 1
            model.Add(score_task[task1] == score_task[task2])

            # way 2
            for mode1, mode2 in modes_allowed_assignment[(task1, task2)]:
                model.AddAllowedAssignments(
                    [is_present_var[(task1, mode1)], is_present_var[(task2, mode2)]],
                    [(1, 1), (0, 0)],
                )

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
                    <= resource_capacity_var[resource]
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
                if capacity > 1 or (
                    capacity == 1 and not use_overlap_for_disjunctive_resource
                ):
                    model.AddCumulative(
                        [interval_var[x[0]] for x in task_modes_consuming]
                        + [x[0] for x in fake_task_res],
                        demands=[x[1] for x in task_modes_consuming]
                        + [x[1] for x in fake_task_res],
                        capacity=self.problem.get_max_resource_capacity(resource),
                    )
                    # We need to add 2
                    model.AddCumulative(
                        [interval_var[x[0]] for x in task_modes_consuming],
                        demands=[x[1] for x in task_modes_consuming],
                        capacity=resource_capacity_var[resource],
                    )
                    for x in task_modes_consuming:
                        model.Add(
                            resource_capacity_var[resource]
                            >= x[1] * is_present_var[x[0]]
                        )
                elif capacity == 1:
                    model.AddNoOverlap(
                        [interval_var[x[0]] for x in task_modes_consuming]
                        + [x[0] for x in fake_task_res]
                    )
                    model.AddMaxEquality(
                        resource_capacity_var[resource],
                        [is_present_var[x[0]] for x in task_modes_consuming],
                    )
        model.Minimize(
            weight_on_used_resource
            * sum([resource_capacity_var[x] for x in resource_capacity_var])
            + weight_on_makespan * starts_var[self.problem.sink_task]
        )
        self.cp_model = model
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "resource_capacity": resource_capacity_var,
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
        # solver.parameters.num_workers = parameters_cp.nb_process
        callback = VarArrayAndObjectiveSolutionPrinter(
            variables=list(self.variables["resource_capacity"].values())
            + [self.variables["start"][self.problem.sink_task]]
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
