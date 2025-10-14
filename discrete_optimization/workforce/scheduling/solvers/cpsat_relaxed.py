#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional, Union

import numpy as np
from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import StatusSolver, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import (
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

    not_implemented_objectives = [
        ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
        ObjectivesEnum.DISPERSION,
        ObjectivesEnum.MIN_WORKLOAD,
    ]

    problem: AllocSchedulingProblem
    variables: dict[str, dict[Any, Any]]

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
        schedule_per_team = {i: [] for i in range(self.problem.number_teams)}
        logger.info(f"Objs = {[cpsolvercb.Value(x) for x in self.variables['objs']]}")
        logger.info(
            f"Obj = {cpsolvercb.ObjectiveValue()}, Bound={cpsolvercb.BestObjectiveBound()}"
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
                        logger.debug("pool ", pool)
                        logger.debug(schedule[i_task, 0], schedule[i_task, 1])
                        logger.debug("Problem with task ", i_task)
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
                                            logger.debug(j)
                                            allocation[j] = team
                                            schedule_per_team[team].append(
                                                (schedule[j, 0], schedule[j, 1])
                                            )
                        if scheduled:
                            break
                    if scheduled:
                        break
                if not scheduled:
                    logger.debug("Still !!")
                    logger.debug(schedule[i_task, 0], schedule[i_task, 1])
                    logger.debug("Problem with task ", i_task)
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

            if "objectives" in self.variables:
                if ObjectivesEnum.MAKESPAN in self.variables["objectives"]:
                    self.cp_model.AddHint(
                        self.variables["objectives"][ObjectivesEnum.MAKESPAN],
                        int(np.max(solution.schedule[:, 1])),
                    )

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
            "optobjectives": {},
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
            self.variables["resource_pool_capacity_var"][i_pool] = (
                self.cp_model.NewIntVar(
                    lb=0, ub=capacity, name=f"capacity_pool_{i_pool}"
                )
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

    def init_model(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **args: Any
    ) -> None:
        if objectives is None:
            objectives = [ObjectivesEnum.NB_TEAMS]
        else:
            for obj in self.not_implemented_objectives:
                if obj in objectives:
                    raise NotImplementedError(
                        f"{obj} not implemented for CPSatAllocSchedulingSolverCumulative"
                    )
        args = self.complete_with_default_hyperparameters(args)
        optional_activities = args["optional_activities"]
        adding_redundant_cumulative = args["adding_redundant_cumulative"]
        add_lower_bound = args["add_lower_bound"]
        super().init_model(**args)
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
            elif obj == ObjectivesEnum.NB_TEAMS:
                objs.append(self.variables["objectives"][ObjectivesEnum.NB_TEAMS])
                weights.append(10000.0)
            elif obj == ObjectivesEnum.MAKESPAN:
                objs.append(self.variables["objectives"][ObjectivesEnum.MAKESPAN])
                weights.append(1.0)
        self.variables["objs"] = objs
        self.cp_model.Minimize(sum(weights[i] * objs[i] for i in range(len(objs))))

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        ortools_cpsat_solver_kwargs: Optional[dict[str, Any]] = None,
        retrieve_stats: bool = False,
        kwargs_scheduling: Optional[dict[str, Any]] = None,
        kwargs_allocation: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Solve the problem with a CpSat scheduling solver chained to a cpsat allocation solver

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_cp: used by subsolvers if not defined in kwargs_scheduling or kwargs_allocation
            time_limit: used by subsolvers if not defined in kwargs_scheduling or kwargs_allocation
            ortools_cpsat_solver_kwargs: used by subsolvers if not defined in kwargs_scheduling or kwargs_allocation
            retrieve_stats: used by subsolvers if not defined in kwargs_scheduling or kwargs_allocation
            kwargs_scheduling: kwargs passed to scheduling solver's `solve()` (including parameters_cp, callbacks, ...)
            kwargs_allocation: kwargs passed to allocation solver's `solve()` (including parameters_cp, callbacks, ...)
            **kwargs: passed to both subsolvers but params are overriden by the one in kwargs_scheduling or kwargs_allocation

        Returns:

        """
        kwargs_allocation = _update_kwargs_subsolver(
            kwargs_subsolver=kwargs_allocation,
            parameters_cp=parameters_cp,
            time_limit=time_limit,
            ortools_cpsat_solver_kwargs=ortools_cpsat_solver_kwargs,
            retrieve_stats=retrieve_stats,
            **kwargs,
        )
        kwargs_scheduling = _update_kwargs_subsolver(
            kwargs_subsolver=kwargs_scheduling,
            parameters_cp=parameters_cp,
            time_limit=time_limit,
            ortools_cpsat_solver_kwargs=ortools_cpsat_solver_kwargs,
            retrieve_stats=retrieve_stats,
            **kwargs,
        )

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        ## step 1: solve with scheduling solver
        res = super().solve(
            **kwargs_scheduling,
        )
        sol1: AllocSchedulingSolution = res[-1][0]
        callbacks_list.on_step_end(step=1, res=res, solver=self)

        ## step 2: solve with allocation solver
        allocation_problem = build_allocation_problem_from_scheduling(
            problem=self.problem, solution=sol1
        )
        allocation_solver = CpsatTeamAllocationSolver(problem=allocation_problem)
        allocation_solver.init_model(
            modelisation_allocation=ModelisationAllocationOrtools.BINARY
        )
        res_ = allocation_solver.solve(**kwargs_allocation)
        if allocation_solver.status_solver == StatusSolver.UNSATISFIABLE:
            self.status_solver = allocation_solver.status_solver
            sol_fits = []
        else:
            sol: TeamAllocationSolution = res_[-1][0]
            rebuilt_solution: AllocSchedulingSolution = sol1
            rebuilt_solution.allocation = np.array(
                [
                    self.problem.teams_to_index[
                        allocation_problem.teams_name[sol.allocation[i]]
                    ]
                    for i in range(len(sol.allocation))
                ]
            )
            fit = self.aggreg_from_sol(rebuilt_solution)
            sol_fits = [(rebuilt_solution, fit)]

        res = self.create_result_storage(sol_fits)
        callbacks_list.on_step_end(step=2, res=res, solver=self)

        callbacks_list.on_solve_end(res=res, solver=self)
        return res


def _update_kwargs_subsolver(
    kwargs_subsolver: Optional[dict[str, Any]] = None,
    callbacks: Optional[list[Callback]] = None,
    parameters_cp: Optional[ParametersCp] = None,
    time_limit: Optional[float] = 100.0,
    ortools_cpsat_solver_kwargs: Optional[dict[str, Any]] = None,
    retrieve_stats: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    updated_kwargs = dict(
        callbacks=callbacks,
        parameters_cp=parameters_cp,
        time_limit=time_limit,
        ortools_cpsat_solver_kwargs=ortools_cpsat_solver_kwargs,
        retrieve_stats=retrieve_stats,
        **kwargs,
    )
    if kwargs_subsolver is not None:
        updated_kwargs.update(kwargs_subsolver)
    return updated_kwargs


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
