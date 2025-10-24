#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from collections.abc import Hashable, Iterable
from enum import Enum
from typing import Any, Optional

from cpmpy.expressions.variables import NDVarArray
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import (
    CpSolver,
    CpSolverSolutionCallback,
    Domain,
    IntVar,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationBinaryOrIntegerModellingCpSatSolver,
    AllocationModelling,
)
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatCallback
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.allocation.problem import (
    AllocationAdditionalConstraint,
    Task,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
    UnaryResource,
)
from discrete_optimization.workforce.allocation.solvers import TeamAllocationSolver
from discrete_optimization.workforce.allocation.utils import (
    compute_all_overlapping,
    compute_equivalent_teams,
)
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.commons.fairness_modeling_ortools import (
    model_fairness,
)


class ModelisationAllocationOrtools(Enum):
    INTEGER = 0
    BINARY = 1
    BINARY_OPTIONAL_ACTIVITIES = 2


class ModelisationDispersionOrtools(Enum):
    EXACT_NAIVE = 0
    # nb teams used given
    EPSILON_TO_AVG_V0 = 1
    # nb teams used not given
    EPSILON_TO_AVG_V1 = 2


logger = logging.getLogger(__name__)


class CpsatTeamAllocationSolver(
    AllocationBinaryOrIntegerModellingCpSatSolver[Task, UnaryResource],
    TeamAllocationSolver,
    WarmstartMixin,
):
    problem: TeamAllocationProblem
    hyperparameters = [
        EnumHyperparameter(
            name="modelisation_allocation",
            enum=ModelisationAllocationOrtools,
            default=ModelisationAllocationOrtools.BINARY,
        ),
        EnumHyperparameter(
            name="modelisation_dispersion",
            enum=ModelisationDispersion,
            default=ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION,
            depends_on=(
                "modelisation_allocation",
                [
                    ModelisationAllocationOrtools.BINARY,
                    ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
                ],
            ),
        ),
        CategoricalHyperparameter(
            name="include_all_binary_vars",
            choices=[True, False],
            default=False,
            depends_on=(
                "modelisation_allocation",
                [
                    ModelisationAllocationOrtools.BINARY,
                    ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
                ],
            ),
        ),
        CategoricalHyperparameter(
            name="include_pair_overlap", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="overlapping_advanced", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="symmbreak_on_used", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="add_lower_bound_nb_teams", choices=[True, False], default=False
        ),
    ]

    at_most_one_unary_resource_per_task = True

    def __init__(
        self,
        problem: TeamAllocationProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.modelisation_allocation: Optional[ModelisationAllocationOrtools] = None
        self.modelisation_dispersion: Optional[ModelisationDispersion] = None
        self.model_max_and_min: bool = False
        self.variables = {}
        self.key_main_decision_variable = None

    def set_warm_start(self, solution: TeamAllocationSolution) -> None:
        self.cp_model.ClearHints()
        if self.modelisation_allocation in (
            ModelisationAllocationOrtools.BINARY,
            ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
        ):
            variables = self.variables["allocation_binary"]
            for i in range(len(solution.allocation)):
                alloc = solution.allocation[i]
                for j in variables[i]:
                    if j == alloc:
                        self.cp_model.AddHint(variables[i][j], 1)
                    else:
                        self.cp_model.AddHint(variables[i][j], 0)
        if self.modelisation_allocation == ModelisationAllocationOrtools.INTEGER:
            variables = self.variables["allocation"]
            for i in range(len(solution.allocation)):
                alloc = solution.allocation[i]
                if alloc is not None and alloc != -1:
                    self.cp_model.AddHint(variables[i], alloc)
                else:
                    self.cp_model.AddHint(variables[i], 0)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"Objective solver : {cpsolvercb.ObjectiveValue()}, Bound : {cpsolvercb.BestObjectiveBound()}"
        )
        allocation: list[int] = []
        if self.modelisation_allocation == ModelisationAllocationOrtools.INTEGER:
            allocation = [
                int(cpsolvercb.Value(var)) for var in self.variables["allocation"]
            ]
        elif self.modelisation_allocation == ModelisationAllocationOrtools.BINARY:
            allocation = []
            variables = self.variables["allocation_binary"]
            for i in range(len(variables)):
                for j in variables[i]:
                    if cpsolvercb.Value(variables[i][j]) == 1:
                        allocation += [j]
                        break
        elif (
            self.modelisation_allocation
            == ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
        ):
            allocation = [None for _ in range(self.problem.number_of_activity)]
            variables = self.variables["allocation_binary"]
            for i in range(len(variables)):
                for j in variables[i]:
                    if cpsolvercb.Value(variables[i][j]) == 1:
                        allocation[i] = j
                        break
        sol = TeamAllocationSolution(problem=self.problem, allocation=allocation)
        # trick to store internal objectives in the output solution !
        sol._intern_objectives = {
            obj: cpsolvercb.value(self.variables["objs"][obj])
            for obj in self.get_lexico_objectives_available()
        }
        return sol

    def get_binary_allocation_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        i_task = self.problem.index_activities_name[task]
        i_team = self.problem.index_teams_name[unary_resource]
        try:
            return self.variables["allocation_binary"][i_task][i_team]
        except KeyError:
            return 0

    def get_integer_allocation_variable(self, task: Task) -> LinearExprT:
        i_task = self.problem.index_activities_name[task]
        return self.variables["allocation"][i_task]

    def init_model(
        self,
        modelisation_allocation: ModelisationAllocationOrtools = ModelisationAllocationOrtools.BINARY,
        **args,
    ):
        args = self.complete_with_default_hyperparameters(args)
        self.modelisation_allocation = modelisation_allocation
        if modelisation_allocation in [
            ModelisationAllocationOrtools.BINARY,
            ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
        ]:
            self.allocation_modelling = AllocationModelling.BINARY
            args["optional_activities"] = (
                modelisation_allocation
                == ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
            )
            self.init_model_binary(**args)
            self.key_main_decision_variable = "allocation_binary"
            if "base_solution" in args:
                self.create_delta_to_base_solution_binary(
                    base_solution=args["base_solution"],
                    base_problem=args.get(
                        "base_problem", args["base_solution"].problem
                    ),
                )
        else:
            self.allocation_modelling = AllocationModelling.INTEGER
            self.init_model_integer(**args)
            self.key_main_decision_variable = "allocation"
            if "base_solution" in args:
                self.create_delta_to_base_solution_integer(
                    base_solution=args["base_solution"],
                    base_problem=args.get(
                        "base_problem", args["base_solution"].problem
                    ),
                )

        if "allocation_additional_constraint" in args or (
            "allocation_additional_constraint" in self.problem.__dict__
            and self.problem.allocation_additional_constraint is not None
        ):
            self.additional_constraint(
                args.get(
                    "allocation_additional_constraint",
                    self.problem.allocation_additional_constraint,
                )
            )

    def init_model_integer(self, **kwargs):
        super().init_model(**kwargs)
        include_pair_overlap = kwargs["include_pair_overlap"]
        overlapping_advanced = kwargs["overlapping_advanced"]
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        add_lower_bound_nb_teams = kwargs["add_lower_bound_nb_teams"]
        assert include_pair_overlap or overlapping_advanced
        domains_for_task: list[Domain] = []
        # Take into account the allocation constraints directly in domains of variable.
        for i in range(self.problem.number_of_activity):
            activity = self.problem.index_to_activities_name[i]
            domains_for_task.append(
                Domain.FromValues(
                    self.problem.compute_allowed_team_index_for_task(activity)
                )
            )
        allocation = [
            self.cp_model.NewIntVarFromDomain(
                domain=domains_for_task[i], name=f"allocation_{i}"
            )
            for i in range(self.problem.number_of_activity)
        ]
        self.variables["allocation"] = allocation

        if include_pair_overlap:
            for edge in self.problem.graph_activity.edges:
                ind1 = self.problem.index_activities_name[edge[0]]
                ind2 = self.problem.index_activities_name[edge[1]]
                self.cp_model.Add(allocation[ind1] != allocation[ind2])

        used = self.create_used_variables_list()
        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            for group in groups:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    self.cp_model.AddImplication(used[ind2], used[ind1])
                    self.cp_model.Add(used[ind1] >= used[ind2])
        if overlapping_advanced or add_lower_bound_nb_teams:
            set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
            if add_lower_bound_nb_teams:
                max_ = max([len(o) for o in set_overlaps])
                self.cp_model.Add(sum(used) >= max_)
            if overlapping_advanced:
                for overlapping in set_overlaps:
                    self.cp_model.AddAllDifferent(
                        [
                            allocation[self.problem.index_activities_name[ac]]
                            for ac in overlapping
                        ]
                    )
        self.variables["objs"] = {}
        self.variables["objs"]["nb_teams"] = sum(used)
        self.variables["used"] = used
        self.variables["keys_variable_to_log"] = []
        self.cp_model.Minimize(self.variables["objs"]["nb_teams"])

    def init_model_binary(self, **kwargs):
        super().init_model(**kwargs)
        optional_activities = kwargs["optional_activities"]
        include_pair_overlap = kwargs["include_pair_overlap"]
        overlapping_advanced = kwargs["overlapping_advanced"]
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        modelisation_dispersion = kwargs["modelisation_dispersion"]
        add_lower_bound_nb_teams = kwargs["add_lower_bound_nb_teams"]
        include_all_binary_vars = kwargs["include_all_binary_vars"]
        assert include_pair_overlap or overlapping_advanced
        domains_for_task: list[list[int]] = []
        # Take into account the allocation constraints directly in domains of variable.
        for i in range(self.problem.number_of_activity):
            if include_all_binary_vars:
                domains_for_task.append(range(self.problem.number_of_teams))
            else:
                activity = self.problem.index_to_activities_name[i]
                domains_for_task.append(
                    self.problem.compute_allowed_team_index_for_task(activity)
                )
        allocation_binary = [
            {
                j: self.cp_model.NewBoolVar(name=f"allocation_{i}_{j}")
                for j in domains_for_task[i]
            }
            for i in range(self.problem.number_of_activity)
        ]
        # store allocation_binary used by other methods like create_used_variables()
        self.variables["allocation_binary"] = allocation_binary
        if include_all_binary_vars:
            for i in range(self.problem.number_of_activity):
                activity = self.problem.index_to_activities_name[i]
                allowed = self.problem.compute_allowed_team_index_for_task(activity)
                if len(allowed) < self.problem.number_of_teams:
                    forbidden = [
                        i
                        for i in range(self.problem.number_of_teams)
                        if i not in allowed
                    ]
                    for f in forbidden:
                        self.cp_model.Add(allocation_binary[i][f] == 0)
        if optional_activities:
            is_allocated = self.create_is_allocated_variables()
        else:
            for i in range(len(allocation_binary)):
                self.cp_model.AddExactlyOne(
                    [allocation_binary[i][j] for j in allocation_binary[i]]
                )

        if include_pair_overlap:
            for edge in self.problem.graph_activity.edges:
                ind1 = self.problem.index_activities_name[edge[0]]
                ind2 = self.problem.index_activities_name[edge[1]]
                for team in allocation_binary[ind1]:
                    if team in allocation_binary[ind2]:
                        if optional_activities:
                            (
                                self.cp_model.AddBoolOr(
                                    ~allocation_binary[ind1][team],
                                    ~allocation_binary[ind2][team],
                                )
                                .OnlyEnforceIf(is_allocated[ind1])
                                .OnlyEnforceIf(is_allocated[ind2])
                            )
                        self.cp_model.AddForbiddenAssignments(
                            [
                                allocation_binary[ind1][team],
                                allocation_binary[ind2][team],
                            ],
                            [(1, 1)],
                        )
        used = self.create_used_variables_list()

        if overlapping_advanced or add_lower_bound_nb_teams:
            set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
            if add_lower_bound_nb_teams:
                max_ = max([len(o) for o in set_overlaps])
                self.cp_model.Add(sum(used) >= max_)
            if overlapping_advanced:
                for overlapping in set_overlaps:
                    for team in self.problem.index_to_teams_name:
                        self.cp_model.Add(
                            sum(
                                [
                                    allocation_binary[
                                        self.problem.index_activities_name[ac]
                                    ][team]
                                    for ac in overlapping
                                    if team
                                    in allocation_binary[
                                        self.problem.index_activities_name[ac]
                                    ]
                                ]
                            )
                            <= 1
                        )

        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            for group in groups:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    self.cp_model.AddImplication(used[ind2], used[ind1])
                    self.cp_model.Add(used[ind1] >= used[ind2])
        self.variables["objs"] = {}
        self.variables["objs"]["nb_teams"] = self.get_nb_unary_resources_used_variable()
        self.variables["used"] = used

        objectives_expr = []
        keys_variable_to_log = []
        if isinstance(self.problem, TeamAllocationProblemMultiobj):
            for key in self.problem.attributes_cumul_activities:
                objectives_expr += [
                    self.add_multiobj(
                        key_objective=key,
                        allocation_binary=allocation_binary,
                        used=used,
                        modelisation_dispersion=modelisation_dispersion,
                    )
                ]
                if objectives_expr[-1] is not None:
                    self.variables["objs"][key] = objectives_expr[-1]
                if objectives_expr[-1] is not None:
                    keys_variable_to_log += [key]
            objectives_expr = [o for o in objectives_expr if o is not None]

        if optional_activities:
            nb_activities_done = self.get_nb_tasks_done_variable()
            self.variables["objs"]["nb_allocated"] = nb_activities_done
            objectives_expr = [nb_activities_done] + objectives_expr
            self.variables["allocated"] = is_allocated
            self.cp_model.Maximize(nb_activities_done)
        else:
            nb_teams = self.variables["objs"]["nb_teams"]
            objectives_expr = [nb_teams] + objectives_expr
            self.cp_model.Minimize(nb_teams)
        self.variables["objectives_expr"] = objectives_expr
        self.variables["keys_variable_to_log"] = keys_variable_to_log

    def create_is_allocated_variables(self) -> list[IntVar]:
        self.create_done_variables()
        return [self.done_variables[task] for task in self.problem.tasks_list]

    def create_used_variables_list(
        self,
    ) -> list[IntVar]:
        self.create_used_variables()
        return [self.used_variables[team] for team in self.problem.teams_name]

    def additional_constraint(
        self, additional_constraint: AllocationAdditionalConstraint
    ):
        attributes = [
            "same_allocation",
            "all_diff_allocation",
            "forced_allocation",
            "forbidden_allocation",
            "allowed_allocation",
            "disjunction",
            "nb_max_teams",
            "precedences",
        ]
        functions_map = {
            "same_allocation": {
                ModelisationAllocationOrtools.BINARY: adding_same_allocation_constraint_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_same_allocation_constraint_binary,
                ModelisationAllocationOrtools.INTEGER: adding_same_allocation_constraint_integer,
            },
            "all_diff_allocation": {
                ModelisationAllocationOrtools.BINARY: adding_all_diff_allocation_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_all_diff_allocation_binary,
                ModelisationAllocationOrtools.INTEGER: adding_all_diff_allocation_integer,
            },
            "forced_allocation": {
                ModelisationAllocationOrtools.BINARY: adding_forced_allocation_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_forced_allocation_binary,
                ModelisationAllocationOrtools.INTEGER: adding_forced_allocation_integer,
            },
            "forbidden_allocation": {
                ModelisationAllocationOrtools.BINARY: adding_forbidden_allocation_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_forbidden_allocation_binary,
                ModelisationAllocationOrtools.INTEGER: adding_forbidden_allocation_integer,
            },
            "allowed_allocation": {
                ModelisationAllocationOrtools.BINARY: adding_allowed_allocation_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_allowed_allocation_binary,
                ModelisationAllocationOrtools.INTEGER: adding_allowed_allocation_integer,
            },
            "disjunction": {
                ModelisationAllocationOrtools.BINARY: adding_disjunction_binary,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_disjunction_binary,
                ModelisationAllocationOrtools.INTEGER: adding_disjunction_integer,
            },
            "nb_max_teams": {
                ModelisationAllocationOrtools.BINARY: adding_max_nb_teams,
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_max_nb_teams,
                ModelisationAllocationOrtools.INTEGER: adding_max_nb_teams,
            },
            "precedences": {
                ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES: adding_precedence_channeling_constraint
            },
        }
        for key in attributes:
            attr = getattr(additional_constraint, key)
            if attr is None:
                continue
            if key == "nb_max_teams":
                functions_map[key][self.modelisation_allocation](
                    attr,
                    model=self.cp_model,
                    variables=self.variables["used"],
                    problem=self.problem,
                )
            elif key == "precedences":
                if self.modelisation_allocation in functions_map[key]:
                    functions_map[key][self.modelisation_allocation](
                        attr,
                        model=self.cp_model,
                        variables=self.variables["allocated"],
                        problem=self.problem,
                    )
            else:
                functions_map[key][self.modelisation_allocation](
                    attr,
                    model=self.cp_model,
                    variables=self.variables[self.key_main_decision_variable],
                    problem=self.problem,
                )

    def create_delta_to_base_solution_binary(
        self,
        base_solution: TeamAllocationSolution,
        base_problem: Optional[TeamAllocationProblem] = None,
    ):
        if base_problem is None:
            base_problem = self.problem
        common_activities = set(base_problem.activities_name).intersection(
            self.problem.activities_name
        )
        indexes = sorted(
            [self.problem.index_activities_name[x] for x in common_activities]
        )
        self.variables["reallocated"] = {}
        for i in indexes:
            activity = self.problem.index_to_activities_name[i]
            index_in_base_problem = base_problem.index_activities_name[activity]
            alloc_base = base_solution.allocation[index_in_base_problem]
            if alloc_base is not None and alloc_base != -1:
                # Avoid dummy values.
                alloc_in_pb = self.problem.index_teams_name[
                    base_problem.teams_name[alloc_base]
                ]
                if alloc_in_pb in self.variables["allocation_binary"][i]:
                    self.variables["reallocated"][i] = (
                        1 - self.variables["allocation_binary"][i][alloc_in_pb]
                    )
                    # self.cp_model.Add()
        self.variables["objs"]["reallocated"] = sum(
            [self.variables["reallocated"][i] for i in self.variables["reallocated"]]
        )

    def create_delta_to_base_solution_integer(
        self,
        base_solution: TeamAllocationSolution,
        base_problem: Optional[TeamAllocationProblem] = None,
    ):
        if base_problem is None:
            base_problem = self.problem
        common_activities = set(base_problem.activities_name).intersection(
            self.problem.activities_name
        )
        indexes = sorted(
            [self.problem.index_activities_name[x] for x in common_activities]
        )
        self.variables["reallocated"] = {}
        for i in indexes:
            activity = self.problem.index_to_activities_name[i]
            index_in_base_problem = base_problem.index_activities_name[activity]
            alloc_base = base_solution.allocation[index_in_base_problem]
            if alloc_base is not None and alloc_base != -1:
                # Avoid dummy values.
                alloc_in_pb = self.problem.index_teams_name[
                    base_problem.teams_name[alloc_base]
                ]
                self.variables["reallocated"][i] = self.cp_model.NewBoolVar(
                    name=f"realloc_{i}"
                )
                self.cp_model.Add(
                    self.variables["allocation"][i] == alloc_in_pb
                ).OnlyEnforceIf(self.variables["reallocated"][i].Not())
                self.cp_model.Add(
                    self.variables["allocation"][i] != alloc_in_pb
                ).OnlyEnforceIf(self.variables["reallocated"][i])
        self.variables["objs"]["reallocated"] = sum(
            [self.variables["reallocated"][i] for i in self.variables["reallocated"]]
        )

    def add_multiobj(
        self,
        key_objective: str,
        allocation_binary: list[dict[int, NDVarArray]],
        used: NDVarArray,
        modelisation_dispersion: ModelisationDispersion,
        **kwargs,
    ):
        assert isinstance(self.problem, TeamAllocationProblemMultiobj)
        values = [
            int(
                self.problem.attributes_of_activities[key_objective][
                    self.problem.activities_name[i]
                ]
            )
            for i in range(self.problem.number_of_activity)
        ]
        dict_dispersion = model_fairness(
            used_team=self.variables["used"],
            allocation_variables=self.variables["allocation_binary"],
            value_per_task=values,
            modelisation_dispersion=modelisation_dispersion,
            cp_model=self.cp_model,
            number_teams=len(self.variables["used"]),
            name_value=key_objective,
        )
        for key in dict_dispersion:
            if key != "obj":
                self.variables[key] = dict_dispersion[key]
        return dict_dispersion["obj"]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.Add(self.variables["objs"][obj] <= int(value))]

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.Minimize(self.variables["objs"][obj])

    def set_model_obj_aggregated(self, objs_weights: list[tuple[str, float]]):
        self.cp_model.Minimize(
            sum([x[1] * self.variables["objs"][x[0]] for x in objs_weights])
        )

    def get_lexico_objectives_available(self) -> list[str]:
        if self.cp_model is not None:
            return list(self.variables["objs"].keys())
        objs = ["nb_teams"]
        if isinstance(self.problem, TeamAllocationProblemMultiobj):
            for key in self.problem.attributes_cumul_activities:
                objs.append(key)
        return objs

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [
            sol._intern_objectives[obj]
            for sol, fit in res.list_solution_fits
            if obj in sol._intern_objectives
        ]
        return min(values)

    def solve_n_best_solution(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        n_best_solution: int = 100,
        time_limit: Optional[float] = 100.0,
        **kwargs,
    ):
        kwargs["modelisation_allocation"] = (
            ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
        )
        if self.cp_model is None:
            self.init_model(**kwargs)
        if "priority" in kwargs:
            self.cp_model.ClearObjective()
            obj = sum(
                [
                    self.variables["allocated"][i] * kwargs["priority"][i]
                    for i in kwargs["priority"]
                ]
            )
            self.cp_model.Maximize(obj)
            self.variables["objectives_expr"][0] = obj
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        callbacks = callbacks_list.callbacks
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        ortools_callback = OrtoolsCpSatCallback(do_solver=self, callback=callbacks_list)
        status = solver.Solve(self.cp_model, ortools_callback)
        logger.info(f"Status: {solver.StatusName(status)}")
        objective = solver.Value(self.variables["objectives_expr"][0])
        warmstart = [
            {
                j: ortools_callback.Value(self.variables["allocation_binary"][i][j])
                for j in self.variables["allocation_binary"][i]
            }
            for i in range(len(self.variables["allocation_binary"]))
        ]
        value = objective
        self.init_model(**kwargs)
        self.cp_model.ClearObjective()
        if "priority" in kwargs:
            self.cp_model.ClearObjective()
            obj = sum(
                [
                    self.variables["allocated"][i] * kwargs["priority"][i]
                    for i in kwargs["priority"]
                ]
            )
            self.variables["objectives_expr"][0] = obj
        for i in range(len(warmstart)):
            for j in warmstart[i]:
                self.cp_model.add_hint(
                    self.variables["allocation_binary"][i][j], warmstart[i][j]
                )
        self.cp_model.Add(self.variables["objectives_expr"][0] == value)
        all_res = []
        while (len(all_res) < n_best_solution) and (value > 0):
            # Drop the objective constraint
            # Update the objective constraint
            solver = CpSolver()
            solver.parameters.num_workers = 1
            solver.parameters.enumerate_all_solutions = True
            callbacks_list = CallbackList(
                callbacks=callbacks
                + [NbIterationStopper(nb_iteration_max=n_best_solution - len(all_res))]
            )
            ortools_callback = OrtoolsCpSatCallback(
                do_solver=self, callback=callbacks_list
            )
            status = solver.Solve(self.cp_model, ortools_callback)
            logger.info(f"Status: {solver.StatusName(status)}")
            for s in ortools_callback.res.list_solution_fits:
                s[0].obj_ = value
            all_res += ortools_callback.res.list_solution_fits
            # Reduce the progress objective value by 1
            value -= 1
            if len(all_res) < n_best_solution:
                # Have to re-init model unfortunately
                # objective = solver.Value(self.variables["objectives_expr"][0])
                self.init_model(**kwargs)
                self.cp_model.ClearObjective()
                if "priority" in kwargs:
                    self.variables["objectives_expr"][0] = sum(
                        [
                            self.variables["allocated"][i] * kwargs["priority"][i]
                            for i in kwargs["priority"]
                        ]
                    )
                self.cp_model.Add(self.variables["objectives_expr"][0] == value)
        return ResultStorage(
            list_solution_fits=all_res,
            mode_optim=self.params_objective_function.sense_function,
        )

    def compute_task_relaxation_alternatives(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        time_limit_per_iteration: Optional[float] = 10.0,
        **kwargs,
    ):
        kwargs["modelisation_allocation"] = (
            ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
        )
        if self.cp_model is None:
            self.init_model(**kwargs)
        assert (
            self.modelisation_allocation
            == ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
        )
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        blocking_constraints = []
        t_initial = time.perf_counter()
        res_final = self.create_result_storage()
        res_optim = self.create_result_storage()
        # variables = self.variables["allocation_binary"]
        variables_allocated = self.variables["allocated"]
        iter_step = 0
        while time.perf_counter() - t_initial <= time_limit:
            tl = min(
                time_limit - (time.perf_counter() - t_initial), time_limit_per_iteration
            )
            logger.info(f"Time limit={tl:.2f}")
            res = self.solve(parameters_cp=parameters_cp, time_limit=tl, **kwargs)
            if self.status_solver == StatusSolver.UNSATISFIABLE:
                break
            logger.info(f"Status of the solver {self.status_solver}")
            res_final.extend(res_final)
            final_sol = res[-1][0]
            res_optim.append(res[-1])
            # values = [{j: self.solver.Value(variables[i][j])
            #            for j in variables[i]}
            #           for i in range(len(variables))]
            not_done = [
                j
                for j in range(len(variables_allocated))
                if self.solver.Value(variables_allocated[j]) == 0
            ]
            logger.info(f"Number of not done : {len(not_done)}")
            logger.info(f"Indexes of not done : {not_done}")
            # blocking_constraints.append(self.cp_model.Add(LinearExpr.Sum([variables_allocated[x]
            #                                                               for x in not_done]) >= 1))
            # Equivalent
            blocking_constraints.append(
                self.cp_model.AddBoolOr([variables_allocated[x] for x in not_done])
            )
            self.set_warm_start(final_sol)
            stopping = callbacks_list.on_step_end(
                step=iter_step, res=res_final, solver=self
            )
            if stopping:
                break
        callbacks_list.on_solve_end(res=res_final, solver=self)
        return res_final, res_optim

    def compute_sufficient_assumptions(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs,
    ):
        self.init_model(
            modelisation_allocation=ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
            **kwargs,
        )
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        solver.parameters.add_clique_cuts = True
        self.cp_model.Maximize(sum(self.variables["allocated"]))
        self.cp_model.AddAssumptions(self.variables["allocated"])
        status = solver.Solve(self.cp_model)
        from ortools.sat.python.cp_model import INFEASIBLE, MODEL_INVALID

        if status in {MODEL_INVALID, INFEASIBLE}:
            # logger.debug(solver.ResponseProto().sufficient_assumptions_for_infeasibility)
            # logger.debug('SufficientAssumptionsForInfeasibility = '
            #      f'{solver.SufficientAssumptionsForInfeasibility()}')
            # logger.debug("Len of assumption : ", len(solver.SufficientAssumptionsForInfeasibility()))
            for var_index in solver.SufficientAssumptionsForInfeasibility():
                print("Infeasible = ")
                print(var_index, self.cp_model.var_index_to_var_proto(var_index))


def adding_same_allocation_constraint_binary(
    same_allocation: list[set[Hashable]],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    from functools import reduce

    for subset_activities in same_allocation:
        index_activities = [
            problem.index_activities_name[activity] for activity in subset_activities
        ]
        common_teams = reduce(
            lambda x, y: x.intersection(set(variables[y].keys())),
            index_activities,
            set(problem.index_to_teams_name.keys()),
        )
        if len(common_teams) == 0:
            logger.warning(f"Your problem is likely to be badly defined.")
        else:
            for c in common_teams:
                model.AddAllowedAssignments(
                    [variables[ind][c] for ind in index_activities],
                    [
                        tuple([1] * len(index_activities)),
                        tuple([0] * len(index_activities)),
                    ],
                )
            # Redundant :
            for c in common_teams:
                for i in range(len(index_activities) - 1):
                    model.Add(
                        variables[index_activities[i]][c]
                        == variables[index_activities[i + 1]][c]
                    )


def adding_same_allocation_constraint_integer(
    same_allocation: list[set[Hashable]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    from functools import reduce

    for subset_activities in same_allocation:
        index_activities = [
            problem.index_activities_name[activity] for activity in subset_activities
        ]
        common_teams = reduce(
            lambda x, y: x.intersection(
                set(problem.compute_allowed_team_index_for_task(y))
            ),
            index_activities,
            set(problem.index_to_teams_name.keys()),
        )
        if len(common_teams) == 0:
            logger.warning(f"Your problem is likely to be badly defined.")
        else:
            model.AddAllowedAssignments(
                [variables[ind] for ind in index_activities],
                [tuple([c] * len(index_activities)) for c in common_teams],
            )
            # Redundant :
            for i in range(len(index_activities) - 1):
                model.Add(
                    variables[index_activities[i]] == variables[index_activities[i + 1]]
                )


def adding_all_diff_allocation_binary(
    all_diff_allocation: list[set[Hashable]],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for all_diff in all_diff_allocation:
        for ac in all_diff:
            ac = problem.index_activities_name[ac]
            for ac2 in all_diff:
                ac2 = problem.index_activities_name[ac2]
                if ac == ac2:
                    continue
                for c in variables[ac]:
                    if c in variables[ac2]:
                        model.AddAtMostOne([variables[ac][c], variables[ac2][c]])
                        model.AddForbiddenAssignments(
                            [variables[ac][c], variables[ac2][c]],
                            [(1, 1)],
                        )


def adding_all_diff_allocation_integer(
    all_diff_allocation: list[set[Hashable]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    for all_diff in all_diff_allocation:
        model.AddAllDifferent(
            [variables[problem.index_activities_name[ac]] for ac in all_diff]
        )


def adding_forced_allocation_binary(
    forced_allocation: dict[Hashable, Hashable],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for ac in forced_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_team = problem.index_teams_name[forced_allocation[ac]]
        if ind_team not in variables[ind_ac]:
            logger.warning(
                "your model is likely invalid, forced allocation not possible"
            )
        else:
            model.Add(variables[ind_ac][ind_team] == 1)
            for team in variables[ind_ac]:
                if team != ind_team:
                    model.Add(variables[ind_ac][team] == 0)

    forbidden_allocation: Optional[dict[Hashable, set[Hashable]]] = (None,)
    disjunction: Optional[list[list[tuple[Hashable, Hashable]]]] = None


def adding_forced_allocation_integer(
    forced_allocation: dict[Hashable, Hashable],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for ac in forced_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_team = problem.index_teams_name[forced_allocation[ac]]
        if ind_team not in problem.compute_allowed_team_index_for_task(ac):
            logger.warning(
                "your model is likely invalid, forced allocation not possible"
            )
        else:
            model.Add(variables[ind_ac] == ind_team)


def adding_forbidden_allocation_binary(
    forbidden_allocation: dict[Hashable, set[Hashable]],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for ac in forbidden_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [problem.index_teams_name[t] for t in forbidden_allocation[ac]]
        model.AddBoolAnd([variables[ind_ac][i].Not() for i in ind_teams])
        # redundant
        model.Add(LinearExpr.Sum([variables[ind_ac][i] for i in ind_teams]) == 0)


def adding_forbidden_allocation_integer(
    forbidden_allocation: dict[Hashable, set[Hashable]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    for ac in forbidden_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [problem.index_teams_name[t] for t in forbidden_allocation[ac]]
        for ind_team in ind_teams:
            model.Add(variables[ind_ac] != ind_team)
        # model.AddBoolAnd([variables[ind_ac] != i
        #                   for i in ind_teams])


def adding_allowed_allocation_binary(
    allowed_allocation: dict[Hashable, set[Hashable]],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for ac in allowed_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [problem.index_teams_name[t] for t in allowed_allocation[ac]]
        for i in variables[ind_ac]:
            if i not in ind_teams:
                model.Add(variables[ind_ac][i] == 0)


def adding_allowed_allocation_integer(
    allowed_allocation: dict[Hashable, set[Hashable]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    for ac in allowed_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [problem.index_teams_name[t] for t in allowed_allocation[ac]]
        model.AddAllowedAssignments([variables[ind_ac]], [(x,) for x in ind_teams])


def adding_disjunction_binary(
    disjunction: list[list[tuple[Hashable, Hashable]]],
    model: cp_model.CpModel,
    variables: list[dict[int, cp_model.IntVar]],
    problem: TeamAllocationProblem,
):
    for disj in disjunction:
        model.AddBoolOr(
            [
                variables[problem.index_activities_name[x[0]]][
                    problem.index_teams_name[x[1]]
                ]
                for x in disj
            ]
        )


def adding_disjunction_integer(
    disjunction: list[list[tuple[Hashable, Hashable]]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    for disj in disjunction:
        bool_vars = []
        for ac, team in disj:
            ind_ac = problem.index_activities_name[ac]
            ind_team = problem.index_teams_name[team]
            bool_var = model.NewBoolVar("")
            model.Add(variables[ind_ac] == ind_team).OnlyEnforceIf(bool_var)
            model.Add(variables[ind_ac] != ind_team).OnlyEnforceIf(bool_var.Not())
            bool_vars.append(bool_var)
        model.AddBoolOr(bool_vars)


def adding_max_nb_teams(
    max_nb_teams: Optional[int],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    if max_nb_teams is not None:
        model.Add(sum(variables) <= max_nb_teams)


def adding_precedence_channeling_constraint(
    precedences: dict[Hashable, set[Hashable]],
    model: cp_model.CpModel,
    variables: list[cp_model.IntVar],
    problem: TeamAllocationProblem,
):
    is_allocated = variables
    if precedences is not None:
        for task in precedences:
            task_index = problem.index_activities_name[task]
            for succ in precedences[task]:
                task_succ = problem.index_activities_name[succ]
                # All equivalent
                model.Add(is_allocated[task_index] == 1).OnlyEnforceIf(
                    is_allocated[task_succ]
                )
                model.Add(is_allocated[task_index] >= is_allocated[task_succ])
                model.AddImplication(is_allocated[task_succ], is_allocated[task_index])
        logger.info("Precedence constraints added")
