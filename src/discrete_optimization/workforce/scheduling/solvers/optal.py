#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional, Union

import numpy as np
import optalcp as cp
from ortools.sat.python.cp_model import IntVar

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import StatusSolver, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    Task,
    UnaryResource,
)
from discrete_optimization.workforce.scheduling.solvers import (
    ObjectivesEnum,
    SolverAllocScheduling,
)
from discrete_optimization.workforce.scheduling.solvers.alloc_scheduling_lb import (
    ApproximateBoundAllocScheduling,
    BaseAllocSchedulingLowerBoundProvider,
    BoundResourceViaRelaxedProblem,
    LBoundAllocScheduling,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    AdditionalCPConstraints,
)
from discrete_optimization.workforce.scheduling.utils import (
    compute_equivalent_teams_scheduling_problem,
)

logger = logging.getLogger(__name__)


class OptalAllocSchedulingSolver(
    SchedulingOptalSolver[Task],
    AllocationOptalSolver[Task, UnaryResource],
    SolverAllocScheduling,
    WarmstartMixin,
):
    hyperparameters = [
        CategoricalHyperparameter(
            name="symmbreak_on_used", choices=[False, True], default=False
        ),
        CategoricalHyperparameter(
            name="optional_activities", choices=[False, True], default=False
        ),
        EnumHyperparameter(
            name="modelisation_dispersion",
            enum=ModelisationDispersion,
            default=ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION,
        ),
        CategoricalHyperparameter(
            name="adding_redundant_cumulative", default=False, choices=[False, True]
        ),
        CategoricalHyperparameter(
            name="add_lower_bound", default=False, choices=[False, True]
        ),
        SubBrickHyperparameter(
            name="lower_bound_method",
            choices=[
                BoundResourceViaRelaxedProblem,
                LBoundAllocScheduling,
                ApproximateBoundAllocScheduling,
            ],
            default=SubBrick(
                BoundResourceViaRelaxedProblem,
                kwargs=BoundResourceViaRelaxedProblem.get_default_hyperparameters(),
            ),
            depends_on=("add_lower_bound", [True]),
        ),
    ]
    problem: AllocSchedulingProblem
    variables: dict[str, dict[Any, Any]]
    cur_sol: cp.Solution

    bound_teams: int = None
    time_bounds: float = 0
    status_bound: StatusSolver = None

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["interval_var"][self.problem.tasks_to_index[task]]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        i_task = self.problem.tasks_to_index[task]
        i_team = self.problem.teams_to_index[unary_resource]
        if i_team in self.variables["opt_interval_var"][i_task]:
            return self.cp_model.presence(
                self.variables["opt_interval_var"][i_task][i_team]
            )
        return 0

    def create_cumul_workload_variables(
        self, values_to_cumul: list, tag: str = "duration"
    ):
        used_variables = self.create_used_variables()
        upper_bound_values = int(sum(values_to_cumul))
        nb_resource = self.problem.number_teams
        workload_per_team = [
            self.cp_model.int_var(
                min=0, max=upper_bound_values, name=f"cumulated_value_{tag}_{i}"
            )
            for i in range(nb_resource)
        ]
        workload_per_team_non_zeros = [
            self.cp_model.int_var(
                min=0, max=upper_bound_values, name=f"cumulated_value_nz_{tag}_{i}"
            )
            for i in range(nb_resource)
        ]
        for index_team in range(nb_resource):
            team = self.problem.unary_resources_list[index_team]
            team_load = self.cp_model.sum(
                [
                    self.get_task_unary_resource_is_present_variable(
                        task=self.problem.tasks_list[i],
                        unary_resource=self.problem.unary_resources_list[index_team],
                    )
                    * values_to_cumul[i]
                    for i in range(self.problem.number_tasks)
                ]
            )
            self.cp_model.enforce(workload_per_team[index_team] == team_load)
            # UNSURE :
            self.cp_model.enforce(
                self.cp_model.implies(
                    used_variables[team],
                    workload_per_team_non_zeros[index_team] == team_load,
                )
            )
            self.cp_model.enforce(
                self.cp_model.implies(
                    self.cp_model.not_(used_variables[team]),
                    workload_per_team_non_zeros[index_team] == upper_bound_values,
                )
            )
        return workload_per_team, workload_per_team_non_zeros

    def add_objective_functions_on_cumul(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args
    ):
        dur = [
            self.problem.tasks_data[t].duration_task for t in self.problem.tasks_list
        ]
        if not (
            ObjectivesEnum.DISPERSION in objectives
            or ObjectivesEnum.MIN_WORKLOAD in objectives
        ):
            return
        d = self.create_cumul_workload_variables(
            values_to_cumul=dur, tag="cumul_workload"
        )
        self.variables["cumul"] = d
        if ObjectivesEnum.DISPERSION in objectives:
            self.variables["min_workload"] = self.cp_model.min(d[1])
            self.variables["max_workload"] = self.cp_model.max(d[0])
            self.variables["objectives"][ObjectivesEnum.DISPERSION] = (
                self.variables["max_workload"] - self.variables["min_workload"]
            )
        if ObjectivesEnum.MIN_WORKLOAD in objectives:
            self.variables["objectives"][ObjectivesEnum.MIN_WORKLOAD] = (
                self.cp_model.min(d[1])
            )

    def set_additional_constraints(
        self, additional_constraint: AdditionalCPConstraints
    ):
        self.set_nb_teams_constraints(additional_constraint=additional_constraint)
        self.set_team_used_constraint(additional_constraint=additional_constraint)

    def set_nb_teams_constraints(self, additional_constraint: AdditionalCPConstraints):
        if additional_constraint.nb_teams_bounds is not None:
            nb_used = self.get_nb_unary_resources_used_variable()
            if (
                additional_constraint.nb_teams_bounds[0] is not None
                and additional_constraint.nb_teams_bounds[1]
                == additional_constraint.nb_teams_bounds[0]
            ):
                self.cp_model.enforce(
                    nb_used == additional_constraint.nb_teams_bounds[0]
                )
            else:
                if additional_constraint.nb_teams_bounds[0] is not None:
                    self.cp_model.enforce(
                        nb_used >= additional_constraint.nb_teams_bounds[0]
                    )
                if additional_constraint.nb_teams_bounds[1] is not None:
                    self.cp_model.enforce(
                        nb_used <= additional_constraint.nb_teams_bounds[1]
                    )

    def set_team_used_constraint(self, additional_constraint: AdditionalCPConstraints):
        if additional_constraint.team_used_constraint is not None:
            used = self.used_variables
            for team_index in additional_constraint.team_used_constraint:
                if additional_constraint.team_used_constraint[team_index] is not None:
                    team = self.problem.unary_resources_list[team_index]
                    # don't care for the syntax warning, i want that it only works with booleans
                    if additional_constraint.team_used_constraint[team_index] == True:
                        self.cp_model.enforce(used[team] == 1)
                    if additional_constraint.team_used_constraint[team_index] == False:
                        self.cp_model.enforce(used[team] == 0)

    def create_delta_objectives(
        self,
        base_solution: AllocSchedulingSolution,
        base_problem: AllocSchedulingProblem,
        additional_constraints: Optional[AdditionalCPConstraints] = None,
    ):
        common_tasks = list(
            set(base_problem.tasks_list).intersection(self.problem.tasks_list)
        )
        len_common_tasks = len(common_tasks)
        reallocation_bool = [
            self.cp_model.bool_var(name=f"realloc_{self.problem.tasks_to_index[t]}")
            for t in common_tasks
        ]
        self.variables["reallocation"] = reallocation_bool
        self.variables["tasks_order_in_reallocation"] = common_tasks
        delta_starts = []
        delta_starts_abs = []
        is_shifted = [
            self.cp_model.bool_var(name=f"shifted_{self.problem.tasks_to_index[t]}")
            for t in common_tasks
        ]
        self.variables["is_shifted"] = is_shifted
        for i in range(len_common_tasks):
            tt = common_tasks[i]
            index_in_problem = self.problem.tasks_to_index[tt]
            ignore_reallocation = False
            if additional_constraints is not None:
                if additional_constraints.set_tasks_ignore_reallocation is not None:
                    if (
                        index_in_problem
                        in additional_constraints.set_tasks_ignore_reallocation
                    ):
                        ignore_reallocation = True

            index_in_base_problem = base_problem.tasks_to_index[tt]
            if (
                base_solution.allocation[index_in_base_problem]
                not in base_problem.index_to_team
            ):
                team_of_base_solution = None
            else:
                team_of_base_solution = base_problem.index_to_team[
                    base_solution.allocation[index_in_base_problem]
                ]

            if team_of_base_solution is not None:
                index_team = self.problem.teams_to_index[team_of_base_solution]
                if not ignore_reallocation:
                    if (
                        index_team
                        not in self.variables["opt_interval_var"][index_in_problem]
                    ):
                        logger.debug(
                            "Problem, original team is not compatible with the task"
                        )
                    else:
                        self.cp_model.enforce(
                            reallocation_bool[i]
                            == self.cp_model.not_(
                                self.get_task_unary_resource_is_present_variable(
                                    task=tt, unary_resource=team_of_base_solution
                                )
                            )
                        )
                else:
                    self.cp_model.enforce(reallocation_bool[i] == 0)

            delta_starts.append(
                -int(base_solution.schedule[index_in_base_problem, 0])
                + self.cp_model.start(self.variables["interval_var"][index_in_problem])
            )
            self.cp_model.enforce(is_shifted[i] == (delta_starts[-1] != 0))
            delta_starts_abs.append(
                self.cp_model.int_var(
                    min=0,
                    max=self.problem.horizon,
                    name=f"delta_abs_starts_{index_in_problem}",
                )
            )
            self.cp_model.enforce(
                delta_starts_abs[-1] == self.cp_model.abs(delta_starts[-1])
            )
        self.variables["delta_starts_abs"] = delta_starts_abs
        self.variables["delta_starts"] = delta_starts
        max_delta_start = self.cp_model.int_var(
            min=0, max=self.problem.horizon, name=f"max_delta_starts"
        )
        self.variables["max_delta_start"] = max_delta_start
        self.cp_model.enforce(max_delta_start == self.cp_model.max(delta_starts_abs))
        objs = [
            sum(reallocation_bool)
        ]  # count the number of changes of team/task allocation
        objs += [sum(delta_starts_abs)]  # sum all absolute shift on the schedule
        objs += [max_delta_start]  # maximum of absolute shift over all tasks
        objs += [
            sum(is_shifted)
        ]  # Number of task that shifted at least by 1 unit of time.
        self.variables["resched_objs"] = {
            "reallocated": objs[0],
            "sum_delta_schedule": objs[1],
            "max_delta_schedule": objs[2],
            "nb_shifted": objs[3],
        }
        return objs

    def create_actually_done_variables(self) -> dict[int, IntVar]:
        if not self.done_variables_created:
            self.done_variables = {}
            for t in self.problem.tasks_list:
                index_task = self.problem.tasks_to_index[t]
                self.done_variables[t] = self.cp_model.sum(
                    [
                        self.cp_model.presence(
                            self.variables["opt_interval_var"][index_task][team]
                        )
                        for team in self.variables["opt_interval_var"][index_task]
                    ]
                )
        return {
            self.problem.tasks_to_index[task]: done_var
            for task, done_var in self.done_variables.items()
        }

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return [
            obj.name if isinstance(obj, ObjectivesEnum) else obj
            for obj in self.variables["objectives"].keys()
        ]

    def set_lexico_objective(self, obj: Union[str, ObjectivesEnum]) -> None:
        obj = _get_variables_obj_key(obj)
        self.cp_model.minimize(self.variables["objectives"][obj])

    def add_lexico_constraint(
        self, obj: Union[str, ObjectivesEnum], value: float
    ) -> Iterable[Any]:
        obj = _get_variables_obj_key(obj)
        return [self.cp_model.enforce(self.variables["objectives"][obj] <= value)]

    def get_lexico_objective_value(
        self, obj: Union[str, ObjectivesEnum], res: ResultStorage
    ) -> float:
        obj = _get_variables_obj_key(obj)
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def set_model_obj_aggregated(
        self, objs_weights: list[tuple[Union[str, ObjectivesEnum], float]]
    ):
        self.cp_model.minimize(
            self.cp_model.sum(
                [
                    x[1] * self.variables["objectives"][_get_variables_obj_key(x[0])]
                    for x in objs_weights
                ]
            )
        )

    def retrieve_solution(self, result: cp.SolutionEvent) -> Solution:
        schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
        allocation = -np.ones(self.problem.number_tasks, dtype=int)
        logger.info(f"Obj = {result.solution.get_objective()}")
        if "resched_objs" in self.variables:
            for obj in self.variables["resched_objs"]:
                logger.info(f"Obj :{obj}")
                logger.info(
                    f"Value : {result.solution.get_value(self.variables['resched_objs'][obj])}"
                )
        for t in range(self.problem.number_tasks):
            start = result.solution.get_start(self.variables["interval_var"][t])
            end = result.solution.get_end(self.variables["interval_var"][t])
            if start is None:
                schedule[t, 0] = self.problem.original_start[self.problem.tasks_list[t]]
            else:
                schedule[t, 0] = start
            if end is None:
                schedule[t, 1] = self.problem.original_end[self.problem.tasks_list[t]]
            else:
                schedule[t, 1] = end
            for index_team in self.variables["opt_interval_var"][t]:
                if result.solution.is_present(
                    self.variables["opt_interval_var"][t][index_team]
                ):
                    allocation[t] = index_team
        sol = AllocSchedulingSolution(
            problem=self.problem, schedule=schedule, allocation=allocation
        )
        sol._intern_obj = {}
        for obj in self.variables["objectives"]:
            sol._intern_obj[obj] = result.solution.get_value(
                self.variables["objectives"][obj]
            )
        self.cur_sol = result.solution
        return sol

    def set_warm_start(self, solution: AllocSchedulingSolution) -> None:
        sol = cp.Solution()
        if solution is not None:
            for t in range(self.problem.number_tasks):
                sol.set_value(
                    self.variables["interval_var"][t],
                    int(solution.schedule[t, 0]),
                    int(solution.schedule[t, 1]),
                )
                for index_team in self.variables["opt_interval_var"][t]:
                    if solution.allocation[t] == index_team:
                        sol.set_value(
                            self.variables["opt_interval_var"][t][index_team],
                            int(solution.schedule[t, 0]),
                            int(solution.schedule[t, 1]),
                        )
                    else:
                        sol.set_absent(
                            self.variables["opt_interval_var"][t][index_team]
                        )
            team_used = set(solution.allocation)
            for team in range(self.problem.number_teams):
                if team in team_used:
                    sol.set_value(self.variables["used"][team], 1)
                else:
                    sol.set_value(self.variables["used"][team], 0)
            self.cur_sol = solution
            self.warm_start_solution = solution
            self.use_warm_start = True

    def init_model(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args: Any
    ) -> None:
        self.cp_model = cp.Model()
        additional_constraints: AdditionalCPConstraints = args.get(
            "additional_constraints", None
        )
        if objectives is None:
            objectives = [ObjectivesEnum.NB_TEAMS]
        args = self.complete_with_default_hyperparameters(args)
        add_lower_bound = args["add_lower_bound"]
        optional_activities = args["optional_activities"]
        adding_redundant_cumulative = args["adding_redundant_cumulative"]
        super().init_model(**args)
        interval_var = {}
        opt_interval_var = {}
        st_lb = [
            (
                int(self.problem.get_lb_start_window(t)),
                int(self.problem.get_ub_start_window(t)),
                int(self.problem.get_lb_end_window(t)),
                int(self.problem.get_ub_end_window(t)),
            )
            for t in self.problem.tasks_list
        ]
        dur = [
            self.problem.tasks_data[t].duration_task for t in self.problem.tasks_list
        ]
        key_per_team = {j: [] for j in self.problem.index_to_team}
        compatible_teams: dict[int, set[int]] = (
            self.problem.compatible_teams_index_all_activity()
        )
        if additional_constraints is not None:
            forced_alloc = additional_constraints.forced_allocation
            if forced_alloc is not None:
                for i in forced_alloc:
                    if forced_alloc[i] is not None:
                        compatible_teams[i] = {forced_alloc[i]}
        for i in range(self.problem.number_tasks):
            interval_var[i] = self.cp_model.interval_var(
                start=(st_lb[i][0], st_lb[i][1]),
                end=(st_lb[i][2], st_lb[i][3]),
                length=dur[i],
                optional=optional_activities,
                name=f"interval_{i}",
            )
            opt_interval_var[i] = {}
            for index_team in compatible_teams[i]:
                opt_interval_var[i][index_team] = self.cp_model.interval_var(
                    start=(st_lb[i][0], st_lb[i][1]),
                    end=(st_lb[i][2], st_lb[i][3]),
                    length=dur[i],
                    optional=True,
                    name=f"interval_{i}_{index_team}",
                )
                key_per_team[index_team].append((i, index_team))
            self.cp_model.alternative(
                interval_var[i],
                [opt_interval_var[i][team] for team in opt_interval_var[i]],
            )
        # Precedence constraints
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                self.cp_model.end_before_start(interval_var[i_t], interval_var[i_t_suc])

        # Same allocation constraints
        for l_t in self.problem.same_allocation:
            indexes = [self.problem.tasks_to_index[tt] for tt in l_t]
            common_teams = reduce(
                lambda x, y: x.intersection(set(opt_interval_var[y].keys())),
                indexes,
                set(self.problem.index_to_team),
            )
            # Redundant
            for c in common_teams:
                for i in range(len(indexes) - 1):
                    self.cp_model.identity(
                        self.cp_model.presence(opt_interval_var[indexes[i]][c]),
                        self.cp_model.presence(opt_interval_var[indexes[i + 1]][c]),
                    )
        # Overlap constraints
        for index_team in key_per_team:
            unavailable = self.problem.compute_unavailability_calendar(
                self.problem.index_to_team[index_team]
            )
            print(unavailable)
            calendar_intervals = [
                self.cp_model.interval_var(start=x[0], end=x[1], length=x[1] - x[0])
                for x in unavailable
            ]
            teams_intervals = [
                opt_interval_var[x[0]][x[1]] for x in key_per_team[index_team]
            ]
            self.cp_model.no_overlap(calendar_intervals + teams_intervals)
            if len(teams_intervals) > 0:
                if additional_constraints is not None:
                    if (
                        additional_constraints.adding_margin_on_sequence[0]
                        and additional_constraints.adding_margin_on_sequence[1] > 0
                    ):
                        margin = additional_constraints.adding_margin_on_sequence[1]
                        seq = self.cp_model.sequence_var(
                            teams_intervals, types=[0] * len(teams_intervals)
                        )
                        seq.no_overlap([[margin]])

        # Resource constraints
        if self.problem.resources_list is not None:
            for resource in self.problem.resources_list:
                capa = int(self.problem.resources_capacity[resource])
                interval_cons = [
                    (
                        interval_var[self.problem.tasks_to_index[t]],
                        self.problem.tasks_data[t].resource_consumption.get(
                            resource, 0
                        ),
                    )
                    for t in self.problem.tasks_data
                    if self.problem.tasks_data[t].resource_consumption.get(resource, 0)
                    > 0
                ]
                if len(interval_cons) > 0:
                    if capa == 1 and all(x[1] == 1 for x in interval_cons):
                        self.cp_model.no_overlap([x[0] for x in interval_cons])
                    else:
                        self.cp_model.enforce(
                            self.cp_model.sum(
                                [self.cp_model.pulse(x[0], x[1]) for x in interval_cons]
                            )
                            <= capa
                        )
        self.variables = {
            "interval_var": interval_var,
            "opt_interval_var": opt_interval_var,
            "objectives": {},
            "key_per_team": key_per_team,
        }
        if optional_activities:
            self.variables["actually_done"] = self.create_actually_done_variables()
        # Objectives definitions
        if ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION in objectives:
            assert "base_solution" in args
            sol: AllocSchedulingSolution = args["base_solution"]
            problem = sol.problem
            self.create_delta_objectives(
                base_solution=sol,
                base_problem=problem,
                additional_constraints=additional_constraints,
            )
        if ObjectivesEnum.MAKESPAN in objectives:
            self.variables["objectives"][ObjectivesEnum.MAKESPAN] = (
                self.get_global_makespan_variable()
            )
        if ObjectivesEnum.NB_DONE_AC in objectives and optional_activities:
            self.variables["objectives"][
                ObjectivesEnum.NB_DONE_AC
            ] = -self.get_nb_tasks_done_variable()
        used = self.create_used_variables()
        self.variables["used"] = used
        if args["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            for group in equivalent_:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    u1 = used[self.problem.unary_resources_list[ind1]]
                    u2 = used[self.problem.unary_resources_list[ind2]]
                    self.cp_model.enforce(self.cp_model.implies(u2, u1))
                    self.cp_model.enforce(u1 >= u2)
        nb_teams_var = self.get_nb_unary_resources_used_variable()
        self.variables["objectives"][ObjectivesEnum.NB_TEAMS] = nb_teams_var
        if adding_redundant_cumulative:
            # we need to introduce this new variable with positive lower bound
            # if we instead directly addCumulative with nb_teams_var, the solver
            # does not find a solution anymore (oddly)
            capacity = self.cp_model.int_var(
                min=1, max=self.problem.number_teams, name="capacity"
            )
            self.cp_model.enforce(capacity == nb_teams_var)
            self.variables["artificial_interval_var"] = self.cp_model.interval_var(
                start=cp.IntervalMin,
                end=cp.IntervalMax,
                optional=False,
                name=f"artificial_interval_var",
            )
            self.cp_model.enforce(
                self.cp_model.sum(
                    [self.cp_model.pulse(interval_var[x], 1) for x in interval_var]
                    + [
                        self.cp_model.pulse(
                            self.variables["artificial_interval_var"],
                            self.problem.number_teams - capacity,
                        )
                    ]
                )
                <= self.problem.number_teams
            )
        if add_lower_bound:
            lprovider: SubBrick = args["lower_bound_method"]
            t_deb = time.perf_counter()
            lbound_provider: BaseAllocSchedulingLowerBoundProvider = lprovider.cls(
                self.problem
            )
            bound = lbound_provider.get_lb_nb_teams(**lprovider.kwargs)
            t_end = time.perf_counter()
            self.cp_model.enforce(nb_teams_var >= bound)
            self.time_bounds = t_end - t_deb
            self.bound_teams = bound
            self.status_bound = lbound_provider.status
        else:
            self.bound_teams = None
            self.time_bounds = 0
            self.status_bound = None

        self.add_objective_functions_on_cumul(objectives=objectives, **args)
        if additional_constraints is not None:
            self.set_additional_constraints(
                additional_constraint=additional_constraints
            )
        objs = []
        weights = []
        for obj in objectives:
            if obj == ObjectivesEnum.NB_DONE_AC and optional_activities:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_DONE_AC])
                weights.append(100000)
            elif obj == ObjectivesEnum.NB_TEAMS:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_TEAMS])
                weights.append(10000)
            elif obj == ObjectivesEnum.MIN_WORKLOAD:
                objs.append(self.variables["objectives"][ObjectivesEnum.MIN_WORKLOAD])
                weights.append(1)
            elif obj == ObjectivesEnum.DISPERSION:
                objs.append(self.variables["objectives"][ObjectivesEnum.DISPERSION])
                weights.append(1)
            elif obj == ObjectivesEnum.MAKESPAN:
                objs.append(self.variables["objectives"][ObjectivesEnum.MAKESPAN])
                weights.append(1)
            elif obj == ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION:
                weights_dict = {
                    "reallocated": 1000,
                    "sum_delta_schedule": 1,
                    "max_delta_schedule": 0,
                    "nb_shifted": 1,
                }
                for x in weights_dict:
                    objs.append(self.variables["resched_objs"][x])
                    weights.append(weights_dict[x])
                    self.variables["objectives"][x] = self.variables["resched_objs"][x]
                self.variables["objectives"][
                    ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION
                ] = sum(
                    [
                        weights_dict[x] * self.variables["resched_objs"][x]
                        for x in weights_dict
                    ]
                )
        for key in self.variables["objectives"]:
            var = self.cp_model.int_var(
                min=cp.IntVarMin, max=cp.IntVarMax, name=f"objectives_{key}"
            )
            self.cp_model.enforce(var == self.variables["objectives"][key])
            self.variables["objectives"][key] = var
        if "resched_objs" in self.variables:
            for key in self.variables["resched_objs"]:
                var = self.cp_model.int_var(
                    min=cp.IntVarMin, max=cp.IntVarMax, name=f"objectives_{key}"
                )
                self.cp_model.enforce(var == self.variables["resched_objs"][key])
                self.variables["resched_objs"][key] = var
        self.variables["objs"] = objs
        self.cp_model.minimize(sum(int(weights[i]) * objs[i] for i in range(len(objs))))


def _get_variables_obj_key(
    obj: Union[str, ObjectivesEnum],
) -> Union[str, ObjectivesEnum]:
    if isinstance(obj, str):
        try:
            return ObjectivesEnum[obj]
        except KeyError:
            return obj
    else:
        return obj
