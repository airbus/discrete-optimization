#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np
from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    IntVar,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NoNonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import NoSkill
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    SinglemodeGenericSchedulingAutoCpSatSolver,
)
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
    NonSkillCumulativeResource,
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


class CPSatAutoAllocSchedulingSolver(
    SinglemodeGenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, NoSkill, NonSkillCumulativeResource, NoNonRenewableResource
    ],
    SolverAllocScheduling,
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
    objective = Objective.NB_UNARY_RESOURCES_USED

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
        return [self.cp_model.add(self.variables["objectives"][obj] <= value)]

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
            sum(
                [
                    x[1] * self.variables["objectives"][_get_variables_obj_key(x[0])]
                    for x in objs_weights
                ]
            )
        )

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, NoSkill]
    ) -> AllocSchedulingSolution:
        schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
        allocation = -np.ones(self.problem.number_tasks, dtype=int)
        for i_task in range(self.problem.number_tasks):
            task = self.problem.index_to_task[i_task]
            task_variable = raw_sol.task_variables[task]
            schedule[i_task, 0] = task_variable.start
            schedule[i_task, 1] = task_variable.end
            for team in task_variable.allocated:
                allocation[i_task] = self.problem.teams_to_index[team]
        sol = AllocSchedulingSolution(
            problem=self.problem, schedule=schedule, allocation=allocation
        )
        sol._intern_obj = raw_sol.metadata
        return sol

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> RawSolution[Task, UnaryResource, NoSkill]:
        """Create solution at temporary format from cpsat variables.

        Override it to
        - add logs
        - store objective variables values in metadata

        """
        # retrieve main tasks variables
        raw_sol = super().retrieve_tasks_variables(cpsolvercb)
        # log objectives
        logger.info(f"Objs = {[cpsolvercb.value(x) for x in self.variables['objs']]}")
        logger.info(
            f"Obj = {cpsolvercb.objective_value}, Bound={cpsolvercb.best_objective_bound}"
        )
        if "resched_objs" in self.variables:
            for obj in self.variables["resched_objs"]:
                logger.info(f"Obj :{obj}")
                logger.info(
                    f"Value : {cpsolvercb.value(self.variables['resched_objs'][obj])}"
                )
        # store objectives
        for obj in self.variables["objectives"]:
            raw_sol.metadata[obj] = cpsolvercb.value(self.variables["objectives"][obj])
        return raw_sol

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        return unary_resource in self.compatible_teams[task]

    def init_model(
        self, objectives: Optional[list[ObjectivesEnum]] = None, **kwargs: Any
    ) -> None:
        # optional parameters
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        additional_constraints: AdditionalCPConstraints = kwargs.get(
            "additional_constraints", None
        )
        # store forced allocation for is_compatible_task_unary_resource
        self.compatible_teams: dict[Task, set[UnaryResource]] = dict(
            self.problem.compatible_teams_per_activity
        )
        if additional_constraints is not None:
            self.compatible_teams.update(
                {
                    self.problem.index_to_task[i_task]: {
                        self.problem.index_to_team[i_team]
                    }
                    for i_task, i_team in additional_constraints.forced_allocation.items()
                }
            )
        self.compatible_tasks: dict[UnaryResource, set[Task]] = {
            team: {
                task
                for task in self.problem.tasks_list
                if team in self.compatible_teams
            }
            for team in self.problem.unary_resources_list
        }

        add_lower_bound = kwargs["add_lower_bound"]
        optional_activities = kwargs["optional_activities"]
        adding_redundant_cumulative = kwargs["adding_redundant_cumulative"]

        self.exactly_one_unary_resource_per_task = not optional_activities

        super().init_model(**kwargs)

        if objectives is None:
            objectives = [ObjectivesEnum.NB_TEAMS]

        self.variables = {}
        self.variables["objectives"] = {}

        # Same allocation constraints
        for common_tasks in self.problem.same_allocation:
            common_teams = [
                team
                for team in self.problem.unary_resources_list
                if all(
                    self.is_compatible_task_unary_resource(
                        task=task, unary_resource=team
                    )
                    for task in common_tasks
                )
            ]

            for team in common_teams:
                self.cp_model.add_allowed_assignments(
                    [
                        self.get_task_unary_resource_is_present_variable(
                            task=task, unary_resource=team
                        )
                        for task in common_tasks
                    ],
                    [
                        tuple([1] * len(common_tasks)),
                        tuple([0] * len(common_tasks)),
                    ],
                )
            # Redundant
            list_common_tasks = list(common_tasks)
            for team in common_teams:
                for i_task in range(len(list_common_tasks) - 1):
                    self.cp_model.add(
                        self.get_task_unary_resource_is_present_variable(
                            task=list_common_tasks[i_task], unary_resource=team
                        )
                        == self.get_task_unary_resource_is_present_variable(
                            task=list_common_tasks[i_task + 1], unary_resource=team
                        )
                    )

        # Overlap constraints
        if additional_constraints is not None:
            for team in self.problem.unary_resources_list:
                if len(self.compatible_tasks[team]) > 0:
                    if (
                        additional_constraints.adding_margin_on_sequence[0]
                        and additional_constraints.adding_margin_on_sequence[1] > 0
                    ):
                        margin = additional_constraints.adding_margin_on_sequence[1]
                        # create just additional interval for the "routing" constraint.
                        if self.avoid_interval_optional:
                            intervals = [
                                self.cp_model.new_fixed_size_interval_var(
                                    start=self.get_task_start_or_end_variable(
                                        task=task, start_or_end=StartOrEnd.START
                                    ),
                                    size=self.problem.tasks_data[task].duration_task
                                    + margin,
                                    name=f"dummy_longer_task_{task, team}",
                                )
                                for task in self.compatible_tasks[team]
                            ]
                            demands = [
                                self.get_task_unary_resource_is_present_variable(
                                    task=task, unary_resource=team
                                )
                                for task in self.compatible_tasks[team]
                            ]
                        else:
                            intervals = [
                                self.cp_model.new_optional_fixed_size_interval_var(
                                    start=self.get_task_start_or_end_variable(
                                        task=task, start_or_end=StartOrEnd.START
                                    ),
                                    size=self.problem.tasks_data[task].duration_task
                                    + margin,
                                    is_present=self.get_task_unary_resource_is_present_variable(
                                        task=task, unary_resource=team
                                    ),
                                    name=f"dummy_longer_task_{task, team}",
                                )
                                for task in self.compatible_tasks[team]
                            ]
                            demands = [1] * len(intervals)
                        self.cp_model.add_cumulative(
                            intervals=intervals,
                            demands=demands,
                            capacity=1,
                        )

        # Objectives definitions
        if ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION in objectives:
            assert "base_solution" in kwargs
            sol: AllocSchedulingSolution = kwargs["base_solution"]
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
        if kwargs["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            for group in equivalent_:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    self.cp_model.add_implication(used[ind2], used[ind1])
                    self.cp_model.add(used[ind1] >= used[ind2])
        nb_teams_var = self.get_nb_unary_resources_used_variable()
        self.variables["objectives"][ObjectivesEnum.NB_TEAMS] = nb_teams_var
        if adding_redundant_cumulative:
            # we need to introduce this new variable with positive lower bound
            # if we instead directly addCumulative with nb_teams_var, the solver
            # does not find a solution anymore (oddly)
            capacity = self.cp_model.new_int_var(
                lb=1, ub=self.problem.number_teams, name="capacity"
            )
            self.cp_model.add(capacity == nb_teams_var)
            self.cp_model.add_cumulative(
                intervals=[
                    self.get_task_interval(task=task)
                    for task in self.problem.tasks_list
                ],
                demands=[1 for _ in range(self.problem.number_tasks)],
                capacity=capacity,
            )
        if add_lower_bound:
            lprovider: SubBrick = kwargs["lower_bound_method"]
            t_deb = time.perf_counter()
            lbound_provider: BaseAllocSchedulingLowerBoundProvider = lprovider.cls(
                self.problem
            )
            bound = lbound_provider.get_lb_nb_teams(**lprovider.kwargs)
            t_end = time.perf_counter()
            self.cp_model.add(nb_teams_var >= bound)
            self.time_bounds = t_end - t_deb
            self.bound_teams = bound
            self.status_bound = lbound_provider.status
        else:
            self.bound_teams = None
            self.time_bounds = 0
            self.status_bound = None

        self.add_objective_functions_on_cumul(objectives=objectives, **kwargs)
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
        self.variables["objs"] = objs
        self.cp_model.minimize(sum(weights[i] * objs[i] for i in range(len(objs))))

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
                    self.allocation_is_present[task] for task in self.problem.tasks_list
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
                    self.allocation_is_present[task] for task in self.problem.tasks_list
                ],
                value_per_task=dur,
                cp_model=self.cp_model,
                number_teams=self.problem.number_teams,
                name_value="workload_",
            )
            min_value = self.cp_model.new_int_var(
                lb=0, ub=sum(dur), name="min_value_workload"
            )

            self.cp_model.add_min_equality(min_value, variables["workload_per_team_nz"])
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
                self.cp_model.add(
                    sum([used[x] for x in used])
                    == additional_constraint.nb_teams_bounds[0]
                )
            else:
                if additional_constraint.nb_teams_bounds[0] is not None:
                    self.cp_model.add(
                        sum([used[x] for x in used])
                        >= additional_constraint.nb_teams_bounds[0]
                    )
                if additional_constraint.nb_teams_bounds[1] is not None:
                    self.cp_model.add(
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
                        self.cp_model.add(used[team_index] == 1)
                    if additional_constraint.team_used_constraint[team_index] == False:
                        self.cp_model.add(used[team_index] == 0)

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
            self.cp_model.new_bool_var(name=f"realloc_{self.problem.tasks_to_index[t]}")
            for t in common_tasks
        ]
        self.variables["reallocation"] = reallocation_bool
        self.variables["tasks_order_in_reallocation"] = common_tasks
        delta_starts = []
        delta_starts_abs = []
        is_shifted = [
            self.cp_model.new_bool_var(name=f"shifted_{self.problem.tasks_to_index[t]}")
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
                if not ignore_reallocation:
                    is_present = self.get_task_unary_resource_is_present_variable(
                        task=tt, unary_resource=team_of_base_solution
                    )
                    self.cp_model.add(is_present == 1).only_enforce_if(
                        reallocation_bool[i].Not()
                    )
                    self.cp_model.add(is_present == 0).only_enforce_if(
                        reallocation_bool[i]
                    )
                else:
                    self.cp_model.add(reallocation_bool[i] == 0)

            delta_starts.append(
                -int(base_solution.schedule[index_in_base_problem, 0])
                + self.get_task_start_or_end_variable(
                    task=tt, start_or_end=StartOrEnd.START
                )
            )
            self.cp_model.add(delta_starts[-1] != 0).only_enforce_if(is_shifted[i])
            self.cp_model.add(delta_starts[-1] == 0).only_enforce_if(
                is_shifted[i].Not()
            )
            delta_starts_abs.append(
                self.cp_model.new_int_var(
                    lb=0,
                    ub=self.problem.horizon,
                    name=f"delta_abs_starts_{index_in_problem}",
                )
            )
            self.cp_model.add_abs_equality(delta_starts_abs[-1], delta_starts[-1])
        self.variables["delta_starts_abs"] = delta_starts_abs
        self.variables["delta_starts"] = delta_starts
        max_delta_start = self.cp_model.new_int_var(
            lb=0, ub=self.problem.horizon, name=f"max_delta_starts"
        )
        self.variables["max_delta_start"] = max_delta_start
        self.cp_model.add_max_equality(max_delta_start, delta_starts_abs)
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
