#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional, Union

import numpy as np
from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar, LinearExprT

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
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
from discrete_optimization.workforce.commons.fairness_modeling_ortools import (
    cumulate_value_per_teams_version_2,
    model_fairness,
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
from discrete_optimization.workforce.scheduling.utils import (
    compute_equivalent_teams_scheduling_problem,
)

logger = logging.getLogger(__name__)


class AdditionalCPConstraints:
    # some object to store additional constraint that could arrive on the way.
    def __init__(
        self,
        nb_teams_bounds: Optional[tuple[Optional[int], Optional[int]]] = None,
        team_used_constraint: Optional[dict[int, bool]] = None,
        set_tasks_ignore_reallocation: Optional[set[int]] = None,
        forced_allocation: Optional[dict[int, int]] = None,
        adding_margin_on_sequence: Optional[tuple[bool, int]] = (False, 0),
    ):
        """
        adding_margin_on_sequence: if the boolean flag is active, we want to add margin between
        end of a task and start of the next one, with value to the integer value of the tuple.
        This trick allow to produce schedule that respects the time transition constraint of the more complex problem
        "AllocSchedRoutingProblem" if the margin is an upper bound of possible transition time.
        """
        self.nb_teams_bounds = nb_teams_bounds
        self.team_used_constraint = team_used_constraint
        self.set_tasks_ignore_reallocation = set_tasks_ignore_reallocation
        self.forced_allocation = forced_allocation
        if nb_teams_bounds is None:
            self.nb_teams_bounds = (None, None)
        if team_used_constraint is None:
            self.team_used_constraint = {}
        if set_tasks_ignore_reallocation is None:
            self.set_tasks_ignore_reallocation = set()
        if forced_allocation is None:
            self.forced_allocation = {}
        self.adding_margin_on_sequence = adding_margin_on_sequence


class CPSatAllocSchedulingSolver(
    SchedulingCpSatSolver[Task],
    AllocationCpSatSolver[Task, UnaryResource],
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

    at_most_one_unary_resource_per_task = True

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        i_task = self.problem.tasks_to_index[task]
        if start_or_end == StartOrEnd.START:
            key = "starts_var"
        else:
            key = "ends_var"
        return self.variables[key][i_task]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable/expression telling if the unary_resource is used for the task.

        NB: sometimes the given resource is never to be used by a task and the variable has not been created.
        The convention is to return 0 in that case.

        """
        i_task = self.problem.tasks_to_index[task]
        i_team = self.problem.teams_to_index[unary_resource]
        try:
            return self.variables["is_present_var"][i_task][i_team]
        except KeyError:
            return 0

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
        self.cp_model.Minimize(self.variables["objectives"][obj])

    def add_lexico_constraint(
        self, obj: Union[str, ObjectivesEnum], value: float
    ) -> Iterable[Any]:
        obj = _get_variables_obj_key(obj)
        return [self.cp_model.Add(self.variables["objectives"][obj] <= value)]

    def get_lexico_objective_value(
        self, obj: Union[str, ObjectivesEnum], res: ResultStorage
    ) -> float:
        obj = _get_variables_obj_key(obj)
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def set_model_obj_aggregated(
        self, objs_weights: list[tuple[Union[str, ObjectivesEnum], float]]
    ):
        self.cp_model.Minimize(
            sum(
                [
                    x[1] * self.variables["objectives"][_get_variables_obj_key(x[0])]
                    for x in objs_weights
                ]
            )
        )

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
        allocation = -np.ones(self.problem.number_tasks, dtype=int)
        logger.info(f"Objs = {[cpsolvercb.Value(x) for x in self.variables['objs']]}")
        logger.info(
            f"Obj = {cpsolvercb.ObjectiveValue()}, Bound={cpsolvercb.BestObjectiveBound()}"
        )
        if "resched_objs" in self.variables:
            for obj in self.variables["resched_objs"]:
                logger.info(f"Obj :{obj}")
                logger.info(
                    f"Value : {cpsolvercb.Value(self.variables['resched_objs'][obj])}"
                )
        for t in range(self.problem.number_tasks):
            start = int(cpsolvercb.Value(self.variables["starts_var"][t]))
            end = int(cpsolvercb.Value(self.variables["ends_var"][t]))
            schedule[t, 0] = start
            schedule[t, 1] = end
            for index_team in self.variables["is_present_var"][t]:
                if cpsolvercb.Value(self.variables["is_present_var"][t][index_team]):
                    allocation[t] = index_team
        sol = AllocSchedulingSolution(
            problem=self.problem, schedule=schedule, allocation=allocation
        )
        sol._intern_obj = {}
        for obj in self.variables["objectives"]:
            sol._intern_obj[obj] = cpsolvercb.Value(self.variables["objectives"][obj])
        return sol

    def set_warm_start(self, solution: AllocSchedulingSolution) -> None:
        if solution is not None:
            self.cp_model.ClearHints()
            for t in range(self.problem.number_tasks):
                self.cp_model.AddHint(
                    self.variables["starts_var"][t], int(solution.schedule[t, 0])
                )
                self.cp_model.AddHint(
                    self.variables["ends_var"][t], int(solution.schedule[t, 1])
                )
                if "actually_done" in self.variables:
                    self.cp_model.AddHint(self.variables["actually_done"][t], 1)

                for index_team in self.variables["is_present_var"][t]:
                    if solution.allocation[t] == index_team:
                        self.cp_model.AddHint(
                            self.variables["is_present_var"][t][index_team], 1
                        )
                    else:
                        self.cp_model.AddHint(
                            self.variables["is_present_var"][t][index_team], 0
                        )
            team_used = set(solution.allocation)
            for team in range(self.problem.number_teams):
                if team in team_used:
                    self.cp_model.AddHint(self.variables["used"][team], 1)
                else:
                    self.cp_model.AddHint(self.variables["used"][team], 0)
            if "reallocation" in self.variables:
                for x in self.variables["reallocation"]:
                    self.cp_model.AddHint(x, 0)
            if "is_shifted" in self.variables:
                for x in self.variables["is_shifted"]:
                    self.cp_model.AddHint(x, 0)
            if "delta_starts_abs" in self.variables:
                for x in self.variables["delta_starts_abs"]:
                    self.cp_model.AddHint(x, 0)
            if "max_delta_start" in self.variables:
                self.cp_model.AddHint(self.variables["max_delta_start"], 0)
            if "objectives" in self.variables:
                if ObjectivesEnum.MAKESPAN in self.variables["objectives"]:
                    self.cp_model.AddHint(
                        self.variables["objectives"][ObjectivesEnum.MAKESPAN],
                        int(np.max(solution.schedule[:, 1])),
                    )

    def init_model(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args: Any
    ) -> None:
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
        starts_var = {}
        ends_var = {}
        is_present_var = {}
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
            starts_var[i] = self.cp_model.NewIntVar(
                lb=st_lb[i][0], ub=st_lb[i][1], name=f"start_{i}"
            )
            ends_var[i] = self.cp_model.NewIntVar(
                lb=st_lb[i][2], ub=st_lb[i][3], name=f"end_{i}"
            )
            interval_var[i] = self.cp_model.NewIntervalVar(
                start=starts_var[i], end=ends_var[i], size=dur[i], name=f"interval_{i}"
            )
            is_present_var[i] = {}
            opt_interval_var[i] = {}
            for index_team in compatible_teams[i]:
                # same as "allocation_binary" variable in the allocation problem
                is_present_var[i][index_team] = self.cp_model.NewBoolVar(
                    name=f"alloc_{i}_{index_team}"
                )
                opt_interval_var[i][index_team] = self.cp_model.NewOptionalIntervalVar(
                    start=starts_var[i],
                    end=ends_var[i],
                    size=dur[i],
                    is_present=is_present_var[i][index_team],
                    name=f"interval_{i}_{index_team}",
                )
                key_per_team[index_team].append((i, index_team))
            if not optional_activities:
                self.cp_model.AddExactlyOne(
                    [is_present_var[i][x] for x in is_present_var[i]]
                )
                # else managed later by self.create_actually_done_variables()
        # Precedence constraints
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                self.cp_model.Add(starts_var[i_t_suc] >= ends_var[i_t])

        # Same allocation constraints
        for l_t in self.problem.same_allocation:
            indexes = [self.problem.tasks_to_index[tt] for tt in l_t]
            common_teams = reduce(
                lambda x, y: x.intersection(set(is_present_var[y].keys())),
                indexes,
                set(self.problem.index_to_team),
            )
            for c in common_teams:
                self.cp_model.AddAllowedAssignments(
                    [is_present_var[ind][c] for ind in indexes],
                    [
                        tuple([1] * len(indexes)),
                        tuple([0] * len(indexes)),
                    ],
                )
            # Redundant
            for c in common_teams:
                for i in range(len(indexes) - 1):
                    self.cp_model.Add(
                        is_present_var[indexes[i]][c]
                        == is_present_var[indexes[i + 1]][c]
                    )

        # Overlap constraints
        for index_team in key_per_team:
            unavailable = self.problem.compute_unavailability_calendar(
                self.problem.index_to_team[index_team]
            )
            fake_tasks_unavailable = [
                self.cp_model.NewFixedSizeIntervalVar(
                    start=x[0], size=x[1] - x[0], name=""
                )
                for x in unavailable
            ]
            tasks_team = [
                opt_interval_var[x[0]][x[1]] for x in key_per_team[index_team]
            ]
            if len(fake_tasks_unavailable) + len(tasks_team) > 0:
                self.cp_model.AddCumulative(
                    tasks_team + fake_tasks_unavailable,
                    [1] * (len(tasks_team) + len(fake_tasks_unavailable)),
                    1,
                )
            if len(tasks_team) > 0:
                if additional_constraints is not None:
                    if (
                        additional_constraints.adding_margin_on_sequence[0]
                        and additional_constraints.adding_margin_on_sequence[1] > 0
                    ):
                        margin = additional_constraints.adding_margin_on_sequence[1]
                        # create just additional interval for the "routing" constraint.
                        intervals = [
                            self.cp_model.NewOptionalFixedSizeIntervalVar(
                                start=starts_var[x[0]],
                                size=dur[x[0]] + margin,
                                is_present=is_present_var[x[0]][x[1]],
                                name=f"dummy_longer_task_{x[0], x[1]}",
                            )
                            for x in key_per_team[index_team]
                        ]
                        self.cp_model.AddCumulative(
                            intervals=intervals,
                            demands=[1] * len(intervals),
                            capacity=1,
                        )

        # Resource constraints
        if self.problem.resources_list is not None:
            for resource in self.problem.resources_list:
                capa = self.problem.resources_capacity[resource]
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
                        self.cp_model.AddNoOverlap([x[0] for x in interval_cons])
                    else:
                        self.cp_model.AddCumulative(
                            intervals=[x[0] for x in interval_cons],
                            demands=[x[1] for x in interval_cons],
                            capacity=capa,
                        )
        self.variables = {
            "starts_var": starts_var,
            "ends_var": ends_var,
            "is_present_var": is_present_var,
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
        used = self.create_used_variables_dict()
        self.variables["used"] = used
        if args["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            for group in equivalent_:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    self.cp_model.AddImplication(used[ind2], used[ind1])
                    self.cp_model.Add(used[ind1] >= used[ind2])
        nb_teams_var = self.get_nb_unary_resources_used_variable()
        self.variables["objectives"][ObjectivesEnum.NB_TEAMS] = nb_teams_var
        if adding_redundant_cumulative:
            # we need to introduce this new variable with positive lower bound
            # if we instead directly addCumulative with nb_teams_var, the solver
            # does not find a solution anymore (oddly)
            capacity = self.cp_model.NewIntVar(
                lb=1, ub=self.problem.number_teams, name="capacity"
            )
            self.cp_model.Add(capacity == nb_teams_var)
            self.cp_model.AddCumulative(
                intervals=[interval_var[x] for x in interval_var],
                demands=[1 for x in interval_var],
                capacity=capacity,
            )
        if add_lower_bound:
            lprovider: SubBrick = args["lower_bound_method"]
            t_deb = time.perf_counter()
            lbound_provider: BaseAllocSchedulingLowerBoundProvider = lprovider.cls(
                self.problem
            )
            bound = lbound_provider.get_lb_nb_teams(**lprovider.kwargs)
            t_end = time.perf_counter()
            self.cp_model.Add(nb_teams_var >= bound)
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
                weights.append(100000.0)
            elif obj == ObjectivesEnum.NB_TEAMS:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_TEAMS])
                weights.append(10000.0)
            elif obj == ObjectivesEnum.MIN_WORKLOAD:
                objs.append(self.variables["objectives"][ObjectivesEnum.MIN_WORKLOAD])
                weights.append(1.0)
            elif obj == ObjectivesEnum.DISPERSION:
                objs.append(self.variables["objectives"][ObjectivesEnum.DISPERSION])
                weights.append(1.0)
            elif obj == ObjectivesEnum.MAKESPAN:
                objs.append(self.variables["objectives"][ObjectivesEnum.MAKESPAN])
                weights.append(1.0)
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
        self.variables["objs"] = objs
        self.cp_model.Minimize(sum(weights[i] * objs[i] for i in range(len(objs))))

    def add_objective_functions_on_cumul(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args
    ):
        modelisation_dispersion: ModelisationDispersion = args[
            "modelisation_dispersion"
        ]
        dur = [
            self.problem.tasks_data[t].duration_task for t in self.problem.tasks_list
        ]
        if ObjectivesEnum.DISPERSION in objectives:
            dict_fairness = model_fairness(
                used_team=self.variables["used"],
                allocation_variables=[
                    self.variables["is_present_var"][i]
                    for i in range(self.problem.number_tasks)
                ],
                value_per_task=dur,
                modelisation_dispersion=modelisation_dispersion,
                cp_model=self.cp_model,
                number_teams=self.problem.number_teams,
                name_value="workload",
            )
            self.variables["cumul_workload"] = dict_fairness["cumulated_value"]
            if "min_value_workload" in dict_fairness:
                self.variables["min_workload"] = dict_fairness["min_value_workload"]
            if "max_value_workload" in dict_fairness:
                self.variables["max_workload"] = dict_fairness["max_value_workload"]
            self.variables["objectives"][ObjectivesEnum.DISPERSION] = dict_fairness[
                "obj"
            ]
        if ObjectivesEnum.MIN_WORKLOAD in objectives:
            variables = cumulate_value_per_teams_version_2(
                used_team=self.variables["used"],
                allocation_variables=[
                    self.variables["is_present_var"][i]
                    for i in range(self.problem.number_tasks)
                ],
                value_per_task=dur,
                cp_model=self.cp_model,
                number_teams=self.problem.number_teams,
                name_value="workload_",
            )
            min_value = self.cp_model.NewIntVar(
                lb=0, ub=sum(dur), name="min_value_workload"
            )

            self.cp_model.AddMinEquality(min_value, variables["workload_per_team_nz"])
            self.variables["objectives"][ObjectivesEnum.MIN_WORKLOAD] = min_value

    def set_additional_constraints(
        self, additional_constraint: AdditionalCPConstraints
    ):
        self.set_nb_teams_constraints(additional_constraint=additional_constraint)
        self.set_team_used_constraint(additional_constraint=additional_constraint)

    def set_nb_teams_constraints(self, additional_constraint: AdditionalCPConstraints):
        if additional_constraint.nb_teams_bounds is not None:
            used = self.variables["used"]
            if (
                additional_constraint.nb_teams_bounds[0] is not None
                and additional_constraint.nb_teams_bounds[1]
                == additional_constraint.nb_teams_bounds[0]
            ):
                self.cp_model.Add(
                    sum([used[x] for x in used])
                    == additional_constraint.nb_teams_bounds[0]
                )
            else:
                if additional_constraint.nb_teams_bounds[0] is not None:
                    self.cp_model.Add(
                        sum([used[x] for x in used])
                        >= additional_constraint.nb_teams_bounds[0]
                    )
                if additional_constraint.nb_teams_bounds[1] is not None:
                    self.cp_model.Add(
                        sum([used[x] for x in used])
                        <= additional_constraint.nb_teams_bounds[1]
                    )

    def set_team_used_constraint(self, additional_constraint: AdditionalCPConstraints):
        if additional_constraint.team_used_constraint is not None:
            used = self.variables["used"]
            for team_index in additional_constraint.team_used_constraint:
                if additional_constraint.team_used_constraint[team_index] is not None:
                    # don't care for the syntax warning, i want that it only works with booleans
                    if additional_constraint.team_used_constraint[team_index] == True:
                        self.cp_model.Add(used[team_index] == 1)
                    if additional_constraint.team_used_constraint[team_index] == False:
                        self.cp_model.Add(used[team_index] == 0)

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
            self.cp_model.NewBoolVar(name=f"realloc_{self.problem.tasks_to_index[t]}")
            for t in common_tasks
        ]
        self.variables["reallocation"] = reallocation_bool
        self.variables["tasks_order_in_reallocation"] = common_tasks
        delta_starts = []
        delta_starts_abs = []
        is_shifted = [
            self.cp_model.NewBoolVar(name=f"shifted_{self.problem.tasks_to_index[t]}")
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
                        not in self.variables["is_present_var"][index_in_problem]
                    ):
                        logger.debug("Problem")
                    else:
                        self.cp_model.Add(
                            self.variables["is_present_var"][index_in_problem][
                                index_team
                            ]
                            == 1
                        ).OnlyEnforceIf(reallocation_bool[i].Not())
                        self.cp_model.Add(
                            self.variables["is_present_var"][index_in_problem][
                                index_team
                            ]
                            == 0
                        ).OnlyEnforceIf(reallocation_bool[i])
                else:
                    self.cp_model.Add(reallocation_bool[i] == 0)

            delta_starts.append(
                -int(base_solution.schedule[index_in_base_problem, 0])
                + self.variables["starts_var"][index_in_problem]
            )
            self.cp_model.Add(delta_starts[-1] != 0).OnlyEnforceIf(is_shifted[i])
            self.cp_model.Add(delta_starts[-1] == 0).OnlyEnforceIf(is_shifted[i].Not())
            delta_starts_abs.append(
                self.cp_model.NewIntVar(
                    lb=0,
                    ub=self.problem.horizon,
                    name=f"delta_abs_starts_{index_in_problem}",
                )
            )
            self.cp_model.AddAbsEquality(delta_starts_abs[-1], delta_starts[-1])
        self.variables["delta_starts_abs"] = delta_starts_abs
        self.variables["delta_starts"] = delta_starts
        max_delta_start = self.cp_model.NewIntVar(
            lb=0, ub=self.problem.horizon, name=f"max_delta_starts"
        )
        self.variables["max_delta_start"] = max_delta_start
        self.cp_model.AddMaxEquality(max_delta_start, delta_starts_abs)
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

    def create_used_variables_dict(
        self,
    ) -> dict[int, IntVar]:
        self.create_used_variables()
        used = {
            self.problem.teams_to_index[team]: used_var
            for team, used_var in self.used_variables.items()
        }
        return used

    def create_actually_done_variables(self) -> dict[int, IntVar]:
        self.create_done_variables()
        return {
            self.problem.tasks_to_index[task]: done_var
            for task, done_var in self.done_variables.items()
        }


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
