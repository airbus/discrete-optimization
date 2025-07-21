#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional

import numpy as np
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import StatusSolver, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import (
    OrtoolsCpSatCallback,
    OrtoolsCpSatSolver,
    ParametersCp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationSolution
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
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
)
from discrete_optimization.workforce.scheduling.solvers import (
    ObjectivesEnum,
    SolverAllocScheduling,
)
from discrete_optimization.workforce.scheduling.solvers.alloc_scheduling_lb import (
    ApproximateBoundAllocScheduling,
    BoundResourceViaRelaxedProblem,
    LBoundAllocScheduling,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    AdditionalCPConstraints,
)
from discrete_optimization.workforce.scheduling.utils import (
    build_allocation_problem_from_scheduling,
    compute_equivalent_teams_scheduling_problem,
    overlap_interval,
)

logger = logging.getLogger(__name__)


class CPSatAllocSchedulingSolverCumulative(
    OrtoolsCpSatSolver, SolverAllocScheduling, WarmstartMixin
):
    hyperparameters = [
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
            depends_on=("add_lower_bound", True),
        ),
    ]

    problem: AllocSchedulingProblem
    variables: dict[str, dict[Any, Any]]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.Minimize(self.variables["objectives"][obj])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.Add(self.variables["objectives"][obj] <= value)]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def set_model_obj_aggregated(self, objs_weights: list[tuple[str, float]]):
        self.cp_model.Minimize(
            sum([x[1] * self.variables["objectives"][x[0]] for x in objs_weights])
        )

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
        allocation = -np.ones(self.problem.number_tasks, dtype=int)
        schedule_per_team = {i: [] for i in range(self.problem.number_teams)}
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

        ordered_task = np.argsort(schedule[:, 0])
        for i_task in ordered_task:
            if allocation[i_task] != -1:
                for l_t in self.problem.same_allocation:
                    indexes = [self.problem.tasks_to_index[tt] for tt in l_t]
                    if i_task in indexes:
                        for j in indexes:
                            if allocation[j] == -1:
                                allocation[j] = allocation[i_task]
                                schedule_per_team[allocation[i_task]].append(
                                    (schedule[j, 0], schedule[j, 1])
                                )
                continue
            scheduled = False
            for mode in self.variables["modes_var"][i_task]:
                if cpsolvercb.Value(self.variables["modes_var"][i_task][mode]):
                    pool = self.resource_pools[mode]
                    for team in pool:
                        if not any(
                            overlap_interval(
                                (schedule[i_task, 0], schedule[i_task, 1]), x
                            )
                            for x in schedule_per_team[team]
                        ):
                            allocation[i_task] = team
                            scheduled = True
                            schedule_per_team[team].append(
                                (schedule[i_task, 0], schedule[i_task, 1])
                            )
                            for l_t in self.problem.same_allocation:
                                indexes = [
                                    self.problem.tasks_to_index[tt] for tt in l_t
                                ]
                                if i_task in indexes:
                                    for j in indexes:
                                        if allocation[j] == -1:
                                            allocation[j] = team
                                            schedule_per_team[team].append(
                                                (schedule[j, 0], schedule[j, 1])
                                            )
                        if scheduled:
                            break
                    if not scheduled:
                        print("pool ", pool)
                        print(schedule[i_task, 0], schedule[i_task, 1])
                        print("Problem with task ", i_task)
            if not scheduled:
                for mode in self.variables["modes_var"][i_task]:
                    pool = self.resource_pools[mode]
                    for team in pool:
                        if not any(
                            overlap_interval(
                                (schedule[i_task, 0], schedule[i_task, 1]), x
                            )
                            for x in schedule_per_team[team]
                        ):
                            allocation[i_task] = team
                            scheduled = True
                            schedule_per_team[team].append(
                                (schedule[i_task, 0], schedule[i_task, 1])
                            )
                            for l_t in self.problem.same_allocation:
                                indexes = [
                                    self.problem.tasks_to_index[tt] for tt in l_t
                                ]
                                if i_task in indexes:
                                    for j in indexes:
                                        if allocation[j] == -1:
                                            print(j)
                                            allocation[j] = team
                                            schedule_per_team[team].append(
                                                (schedule[j, 0], schedule[j, 1])
                                            )
                        if scheduled:
                            break
                    if scheduled:
                        break
                if not scheduled:
                    print("Still !!")
                    print(schedule[i_task, 0], schedule[i_task, 1])
                    print("Problem with task ", i_task)
        # if np.sum(allocation == -1) > 0:
        #     return None
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
                # if ObjectivesEnum.NB_DONE_AC in self.variables["objectives"]:
                #     self.cp_model.AddHint(self.variables["objectives"][ObjectivesEnum.NB_DONE_AC],
                #                           len([x for x in solution.allocation
                #                                if not np.isnan(x) and x != 1]))
        if self.solver is not None:
            self.cp_model.ClearHints()
            response = self.solver.ResponseProto()  # Get the raw response
            for i in range(len(response.solution)):
                var = self.cp_model.GetIntVarFromProtoIndex(i)
                # print(f"Variable {var} = {response.solution[i]}")
                self.cp_model.AddHint(var, response.solution[i])

    def init_multimode_data(self, **kwargs):
        equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
        compatible_teams = self.problem.compatible_teams_index_all_activity()
        self.compatible_teams = compatible_teams
        self.resource_pools = equivalent_
        self.available_resource_per_task = {}
        for i in range(self.problem.number_tasks):
            self.available_resource_per_task[i] = set()
            comp_team = compatible_teams[i]
            for team_index in comp_team:
                for i_r in range(len(self.resource_pools)):
                    if team_index in self.resource_pools[i_r]:
                        self.available_resource_per_task[i].add(i_r)
                        assert all(
                            x in compatible_teams[i] for x in self.resource_pools[i_r]
                        )  # Normally, all resource in this resource pool is able to do this task

    def init_main_vars(self, **args):
        self.init_multimode_data(**args)
        optional_activities = args["optional_activities"]
        starts_var = {}
        ends_var = {}
        modes_var = {}
        interval_var = {}
        actually_done_var = {}
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
        keys_per_mode = {m: [] for m in range(len(self.resource_pools))}
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
            opt_interval_var[i] = {}
            modes_var[i] = {}
            if optional_activities:
                actually_done_var[i] = self.cp_model.NewBoolVar(name=f"done_{i}")
            for mode in self.available_resource_per_task[i]:
                modes_var[i][mode] = self.cp_model.NewBoolVar(name=f"task_{i}_{mode}")
                opt_interval_var[i][mode] = self.cp_model.NewOptionalIntervalVar(
                    start=starts_var[i],
                    end=ends_var[i],
                    size=dur[i],
                    is_present=modes_var[i][mode],
                    name=f"interval_{i}_{mode}",
                )
                keys_per_mode[mode].append((i, mode))
            if optional_activities:
                self.cp_model.AddBoolOr(
                    [modes_var[i][mode] for mode in modes_var[i]]
                ).OnlyEnforceIf(actually_done_var[i])
            else:
                self.cp_model.AddExactlyOne(
                    [modes_var[i][mode] for mode in modes_var[i]]
                )

        self.variables = {
            "starts_var": starts_var,
            "ends_var": ends_var,
            "modes_var": modes_var,
            "interval_var": interval_var,
            "opt_interval_var": opt_interval_var,
            "opt" "objectives": {},
            "key_per_mode": keys_per_mode,
        }
        if optional_activities:
            self.variables["actually_done"] = actually_done_var

    def set_precedence_constraints(self):
        starts_var = self.variables["starts_var"]
        ends_var = self.variables["ends_var"]
        # Precedence constraints
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                self.cp_model.Add(starts_var[i_t_suc] >= ends_var[i_t])

    def set_same_allocation_constraints(self):
        # Same allocation constraints
        is_present_var = self.variables["modes_var"]
        for l_t in self.problem.same_allocation:
            indexes = [self.problem.tasks_to_index[tt] for tt in l_t]
            common_modes = reduce(
                lambda x, y: x.intersection(set(is_present_var[y].keys())),
                indexes,
                set(self.problem.index_to_team),
            )
            for c in common_modes:
                self.cp_model.AddAllowedAssignments(
                    [is_present_var[ind][c] for ind in indexes],
                    [
                        tuple([1] * len(indexes)),
                        tuple([0] * len(indexes)),
                    ],
                )
            # They don't overlap :-)
            self.cp_model.AddNoOverlap(
                [self.variables["interval_var"][x] for x in indexes]
            )
            # Redundant
            for c in common_modes:
                for i in range(len(indexes) - 1):
                    self.cp_model.Add(
                        is_present_var[indexes[i]][c]
                        == is_present_var[indexes[i + 1]][c]
                    )

    def set_resource_pool_constraints(self):
        pools = self.resource_pools
        self.variables["resource_pool_capacity_var"] = {}
        for i_pool in range(len(pools)):
            capacity = len(pools[i_pool])
            self.variables["resource_pool_capacity_var"][
                i_pool
            ] = self.cp_model.NewIntVar(
                lb=0, ub=capacity, name=f"capacity_pool_{i_pool}"
            )
            keys = self.variables["key_per_mode"][i_pool]
            intervals = [self.variables["opt_interval_var"][x[0]][x[1]] for x in keys]
            some_team = pools[i_pool][0]
            unavailable = self.problem.compute_unavailability_calendar(
                self.problem.index_to_team[some_team]
            )
            fake_tasks_unavailable = [
                self.cp_model.NewFixedSizeIntervalVar(
                    start=x[0], size=x[1] - x[0], name=""
                )
                for x in unavailable
            ]
            self.cp_model.AddCumulative(
                intervals=intervals + fake_tasks_unavailable,
                demands=[1] * len(intervals) + [capacity] * len(fake_tasks_unavailable),
                capacity=capacity,
            )
            self.cp_model.AddCumulative(
                intervals=intervals,
                demands=[1] * len(intervals),
                capacity=self.variables["resource_pool_capacity_var"][i_pool],
            )

    def set_resource_constraints(self):
        interval_var = self.variables["interval_var"]
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

    def add_buffers(self):
        pass

    #
    #
    #
    #
    # if len(tasks_team) > 0:
    #     if additional_constraints is not None:
    #         if (additional_constraints.adding_margin_on_sequence[0]
    #                 and additional_constraints.adding_margin_on_sequence[1] > 0):
    #             margin = additional_constraints.adding_margin_on_sequence[1]
    #             # create just additional interval for the "routing" constraint.
    #             intervals = [
    #                 self.cp_model.NewOptionalFixedSizeIntervalVar(
    #                     start=starts_var[x[0]],
    #                     size=dur[x[0]]+margin,
    #                     is_present=is_present_var[x[0]][x[1]],
    #                     name=f"dummy_longer_task_{x[0],x[1]}"
    #                 )
    #                 for x in key_per_team[index_team]
    #             ]
    #             self.cp_model.AddCumulative(intervals=intervals,
    #                                         demands=[1]*len(intervals),
    #                                         capacity=1)

    def create_makespan_obj(
        self, ends_var: dict[int, IntVar], st_lb: list[tuple[int, int, int, int]] = None
    ):
        if st_lb is None:
            st_lb = [
                (
                    int(self.problem.get_lb_start_window(t)),
                    int(self.problem.get_ub_start_window(t)),
                    int(self.problem.get_lb_end_window(t)),
                    int(self.problem.get_ub_end_window(t)),
                )
                for t in self.problem.tasks_list
            ]
        lb_makespan = max([x[2] for x in st_lb])
        ub_makespan = max([x[3] for x in st_lb])
        makespan = self.cp_model.NewIntVar(
            lb=lb_makespan, ub=ub_makespan, name="makespan"
        )
        self.cp_model.AddMaxEquality(makespan, [ends_var[i] for i in ends_var])
        return makespan

    def define_objectives(self, objectives: list[ObjectivesEnum], **args):
        self.variables["objectives"] = {}
        optional_activities = args["optional_activities"]
        if args["optional_activities"]:
            actually_done_var = self.variables["actually_done"]
        ends_var = self.variables["ends_var"]
        if ObjectivesEnum.MAKESPAN in objectives:
            makespan = self.create_makespan_obj(ends_var, None)
            self.variables["objectives"][ObjectivesEnum.MAKESPAN] = makespan
        if ObjectivesEnum.NB_DONE_AC in objectives and optional_activities:
            nb_done = sum([actually_done_var[i] for i in actually_done_var])
            self.variables["objectives"][ObjectivesEnum.NB_DONE_AC] = -nb_done
        if ObjectivesEnum.NB_TEAMS in objectives:
            self.variables["objectives"][ObjectivesEnum.NB_TEAMS] = sum(
                [
                    self.variables["resource_pool_capacity_var"][x]
                    for x in self.variables["resource_pool_capacity_var"]
                ]
            )
        self.add_objective_functions_on_cumul(objectives=objectives, **args)

    def init_model(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args: Any
    ) -> None:
        additional_constraints: AdditionalCPConstraints = args.get(
            "additional_constraints", None
        )
        if objectives is None:
            objectives = [ObjectivesEnum.NB_TEAMS]
        args = self.complete_with_default_hyperparameters(args)
        optional_activities = args["optional_activities"]
        adding_redundant_cumulative = args["adding_redundant_cumulative"]
        add_lower_bound = args["add_lower_bound"]
        self.cp_model = cp_model.CpModel()
        self.init_main_vars(**args)
        self.set_precedence_constraints()
        self.set_same_allocation_constraints()
        self.set_resource_pool_constraints()
        self.define_objectives(objectives=objectives, **args)
        if adding_redundant_cumulative:
            interval_var = self.variables["interval_var"]
            sum_capa = sum(self.variables["resource_pool_capacity_var"].values())
            capa = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.number_teams, name=f"full_capa"
            )
            self.cp_model.Add(capa == sum_capa)
            self.cp_model.AddCumulative(
                intervals=[interval_var[x] for x in interval_var],
                demands=[1 for x in interval_var],
                capacity=capa,
            )
        if add_lower_bound:
            lprovider: SubBrick = args["lower_bound_method"]
            t_deb = time.perf_counter()
            lbound_provider: BoundResourceViaRelaxedProblem = lprovider.cls(
                self.problem
            )
            bound = lbound_provider.get_lb_nb_teams(**lprovider.kwargs)
            t_end = time.perf_counter()
            self.cp_model.Add(
                sum(self.variables["resource_pool_capacity_var"].values()) >= bound
            )
            self.time_bounds = t_end - t_deb
            self.bound_teams = bound
            self.status_bound = lbound_provider.status
        objs = []
        weights = []
        for obj in objectives:
            if obj == ObjectivesEnum.NB_DONE_AC and optional_activities:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_DONE_AC])
                weights.append(100000.0)
            if obj == ObjectivesEnum.NB_TEAMS:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_TEAMS])
                weights.append(10000.0)
            if obj == ObjectivesEnum.MIN_WORKLOAD:
                objs.append(self.variables["objectives"][ObjectivesEnum.MIN_WORKLOAD])
                weights.append(1.0)
            if obj == ObjectivesEnum.DISPERSION:
                objs.append(self.variables["objectives"][ObjectivesEnum.DISPERSION])
                weights.append(1.0)
            if obj == ObjectivesEnum.DISPERSION_DISTANCE:
                objs.append(
                    self.variables["objectives"][ObjectivesEnum.DISPERSION_DISTANCE]
                )
                weights.append(1.0)
            if obj == ObjectivesEnum.MIN_DISTANCE:
                objs.append(self.variables["objectives"][ObjectivesEnum.MIN_DISTANCE])
                weights.append(1.0)
            if obj == ObjectivesEnum.MAX_DISTANCE:
                objs.append(self.variables["objectives"][ObjectivesEnum.MAX_DISTANCE])
                weights.append(1.0)
            if obj == ObjectivesEnum.MAKESPAN:
                objs.append(self.variables["objectives"][ObjectivesEnum.MAKESPAN])
                weights.append(1.0)
            if obj == ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION:
                weights_dict = {
                    "reallocated": 1000,
                    "sum_delta_schedule": 1,  # 100,
                    "max_delta_schedule": 0,  # 10,
                    "nb_shifted": 1,
                }
                for x in weights_dict:
                    objs.append(self.variables["resched_objs"][x])
                    weights.append(weights_dict[x])
                    self.variables["objectives"][x] = self.variables["resched_objs"][x]
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

    def create_delta_objectives(
        self,
        base_solution: AllocSchedulingSolution,
        base_problem: AllocSchedulingProblem,
        additional_constraints: Optional[AdditionalCPConstraints] = None,
    ):
        objs = []
        common_tasks = list(
            set(base_problem.tasks_list).intersection(self.problem.tasks_list)
        )
        common_teams = list(
            set(base_problem.team_names).intersection(self.problem.team_names)
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
                    # index_in_problem, index_team, i)
                    # print(index_in_problem in self.variables["is_present_var"])
                    # print(index_team in self.variables["is_present_var"][index_in_problem])
                    if (
                        index_team
                        not in self.variables["is_present_var"][index_in_problem]
                    ):
                        print("Problem")
                    else:
                        # print(self.variables["is_present_var"][index_in_problem].keys())
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
        # ask to minimize the maximum abs delta (shift of the schedule)
        # self.cp_model.Minimize(max_delta_start)
        self.variables["resched_objs"] = {
            "reallocated": objs[0],
            "sum_delta_schedule": objs[1],
            "max_delta_schedule": objs[2],
            "nb_shifted": objs[3],
        }
        return objs

    def solve_two_step(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        ortools_cpsat_solver_kwargs: Optional[dict[str, Any]] = None,
        retrieve_stats: bool = False,
    ):
        res = self.solve(
            callbacks=callbacks,
            parameters_cp=parameters_cp,
            time_limit=time_limit,
            ortools_cpsat_solver_kwargs=ortools_cpsat_solver_kwargs,
            retrieve_stats=retrieve_stats,
        )
        allocation_problem = build_allocation_problem_from_scheduling(
            problem=self.problem, solution=res[-1][0]
        )
        allocation_solver = CpsatTeamAllocationSolver(problem=allocation_problem)
        allocation_solver.init_model(
            modelisation_allocation=ModelisationAllocationOrtools.BINARY
        )
        res_ = allocation_solver.solve(
            parameters_cp=parameters_cp,
            time_limit=time_limit,
            ortools_cpsat_solver_kwargs=ortools_cpsat_solver_kwargs,
            retrieve_stats=retrieve_stats,
        )
        if allocation_solver.status_solver == StatusSolver.UNSATISFIABLE:
            self.status_solver = allocation_solver.status_solver
            return self.create_result_storage([])
        else:
            sol: TeamAllocationSolution = res_[-1][0]
            rebuilt_solution: AllocSchedulingSolution = res[-1][0]
            rebuilt_solution.allocation = [
                self.problem.teams_to_index[
                    allocation_problem.teams_name[sol.allocation[i]]
                ]
                for i in range(len(sol.allocation))
            ]
            fit = self.aggreg_from_sol(rebuilt_solution)
            return self.create_result_storage([(rebuilt_solution, fit)])
