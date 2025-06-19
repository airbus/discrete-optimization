#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import inspect
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from time import time
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple, Union

import cpmpy
import cpmpy as cp
from cpmpy import (
    AllDifferent,
    AllEqual,
    Maximum,
    Minimum,
    Model,
    SolverLookup,
    boolvar,
    intvar,
)
from cpmpy.expressions.variables import NDVarArray
from cpmpy.model import Expression
from cpmpy.solvers import CPM_ortools
from cpmpy.solvers.ortools import OrtSolutionCounter
from cpmpy.solvers.solver_interface import ExitStatus, SolverStatus
from cpmpy.transformations.normalize import toplevel_list

from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    ParametersCp,
    StatusSolver,
)
from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.allocation.allocation_problem_utils import (
    compute_all_overlapping,
    compute_equivalent_teams,
)
from discrete_optimization.workforce.allocation.problem import (
    AggregateOperator,
    AllocationAdditionalConstraint,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.solvers import TeamAllocationSolver
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.commons.fairness_modeling_cpmpy import (
    model_fairness,
)

logger = logging.getLogger(__name__)
this_folder = os.path.abspath(os.path.dirname(__file__))
root_folder = os.path.join(this_folder, "../../")
path_translation = os.path.join(
    root_folder, "scheduling_hhcs/app/study_tabs/german_translation.json"
)
if os.path.exists(path_translation):
    with open(path_translation, "r") as file:
        data = file.read()
        language_dict = json.loads(data)
else:
    language_dict = {}


def from_solver_status_cpmpy_to_status_solver(
    solver_status: SolverStatus,
) -> StatusSolver:
    if solver_status.exitstatus == ExitStatus.OPTIMAL:
        return StatusSolver.OPTIMAL
    if solver_status.exitstatus == ExitStatus.FEASIBLE:
        return StatusSolver.SATISFIED
    if solver_status.exitstatus == ExitStatus.UNKNOWN:
        return StatusSolver.UNKNOWN
    if solver_status.exitstatus == ExitStatus.UNSATISFIABLE:
        return StatusSolver.UNSATISFIABLE


class ModelisationAllocationCP(Enum):
    INTEGER = 0
    BINARY = 1
    CNF_COMPATIBLE = 2


@dataclass
class AllowedTeamForTask:
    constraint: cpmpy.model.Expression
    task: Hashable
    teams: Set[Hashable]
    description: str

    def to_dict(self):
        return {"task": self.task, "description": self.description}


@dataclass
class AllocatedTaskConstraint:
    constraint: cpmpy.model.Expression
    task: Hashable
    description: str

    def to_dict(self):
        return {"task": self.task, "description": self.description}


@dataclass
class NoOverlapByTeamConstraint:
    constraint: cpmpy.model.Expression
    tasks: Set[Hashable]
    team: Hashable
    description: str

    def to_dict(self):
        return {
            "tasks": list(self.tasks),
            "team": self.team,
            "description": self.description,
        }


@dataclass
class NoOverlapConstraint:
    """
    Conjonction of NoOverlapByTeamConstraint in practice
    """

    constraint: cpmpy.model.Expression
    tasks: Set[Hashable]
    description: str

    def to_dict(self):
        return {"tasks": list(self.tasks), "description": self.description}


@dataclass
class UsedTeamConstraint:
    constraint: cpmpy.model.Expression
    team: Hashable
    description: str

    def to_dict(self):
        return {"team": list(self.team), "description": self.description}


@dataclass
class SymmetryBreakingConstraint:
    constraint: cpmpy.model.Expression
    teams: Set[Hashable]
    description: str

    def to_dict(self):
        return {"team": list(self.teams), "description": self.description}


@dataclass
class SameAllocationConstraint:
    constraint: cpmpy.model.Expression
    tasks: Set[Hashable]
    description: str

    def to_dict(self):
        return {"tasks": list(self.tasks), "description": self.description}


@dataclass
class ForcedAllocationConstraint:
    constraint: cpmpy.model.Expression
    task: Hashable
    team: Hashable
    description: str

    def to_dict(self):
        return {"task": self.task, "team": self.team, "description": self.description}


@dataclass
class ForbiddenAllocationConstraint:
    constraint: cpmpy.model.Expression
    task: Hashable
    teams: Set[Hashable]
    description: str

    def to_dict(self):
        return {
            "task": self.task,
            "teams": list(self.teams),
            "description": self.description,
        }


@dataclass
class DisjunctionConstraint:
    constraint: cpmpy.model.Expression
    task_team_disjunction: List[Tuple[Hashable, Hashable]]
    description: str

    def to_dict(self):
        return {
            "task_team_disjunction": self.task_team_disjunction,
            "description": self.description,
        }


@dataclass
class MaxNbTeams:
    constraint: cpmpy.model.Expression
    nb_max_teams: int
    description: str

    def to_dict(self):
        return {"nb_max_teams": self.nb_max_teams, "description": self.description}


@dataclass
class TaskChanneling:
    constraint: cpmpy.model.Expression
    # tasks[1] is done -> tasks[0] is done
    tasks: tuple[Hashable, Hashable]
    description: str

    def to_dict(self):
        return {"tasks": self.tasks, "description": self.description}


CONSTRAINT_ABSTRACTION = Union[
    NoOverlapConstraint,
    NoOverlapByTeamConstraint,
    AllocatedTaskConstraint,
    AllowedTeamForTask,
    UsedTeamConstraint,
    SymmetryBreakingConstraint,
    SameAllocationConstraint,
    ForcedAllocationConstraint,
    ForbiddenAllocationConstraint,
    DisjunctionConstraint,
    MaxNbTeams,
]


class CallbackWithBound(OrtSolutionCounter):
    def on_solution_callback(self):
        super().on_solution_callback()
        print(f"Obj bound, {self.BestObjectiveBound()}")


class CPMpyTeamAllocationSolver(CpmpySolver, CpSolver, TeamAllocationSolver):
    problem: TeamAllocationProblem
    hyperparameters = [
        EnumHyperparameter(
            name="modelisation_allocation",
            enum=ModelisationAllocationCP,
            default=ModelisationAllocationCP.BINARY,
        ),
        EnumHyperparameter(
            name="modelisation_dispersion",
            enum=ModelisationDispersion,
            default=ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION,
        ),
        CategoricalHyperparameter(
            name="include_all_binary_vars", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="include_pair_overlap", choices=[True, False], default=False
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

    def __init__(
        self,
        problem: TeamAllocationProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        solver_name: Optional[str] = None,
        **kwargs: Any,
    ):
        CpSolver.__init__(self, problem, params_objective_function, **kwargs)
        self.model: Optional[Model] = None
        self.solver_name = solver_name
        self.cpm_solver = None
        self.modelisation_allocation: Optional[ModelisationAllocationCP] = None
        self.variables = {}
        self.key_main_decision_variable: Optional[str] = None
        self.meta_constraints = []

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
                ModelisationAllocationCP.BINARY: adding_same_allocation_constraint_binary,
                ModelisationAllocationCP.INTEGER: adding_same_allocation_constraint_integer,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_same_allocation_constraint_cnf,
            },
            "all_diff_allocation": {
                ModelisationAllocationCP.BINARY: adding_all_diff_allocation_binary,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_all_diff_allocation_binary,
            },
            "forced_allocation": {
                ModelisationAllocationCP.BINARY: adding_forced_allocation_binary,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_forced_allocation_binary,
            },
            "forbidden_allocation": {
                ModelisationAllocationCP.BINARY: adding_forbidden_allocation_binary,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_forbidden_allocation_binary,
            },
            "allowed_allocation": {
                ModelisationAllocationCP.BINARY: adding_allowed_allocation_binary,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_allowed_allocation_binary,
            },
            "disjunction": {
                ModelisationAllocationCP.BINARY: adding_disjunction_binary,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_disjunction_binary,
            },
            "nb_max_teams": {
                ModelisationAllocationCP.BINARY: adding_max_nb_teams,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_max_nb_teams,
            },
            "precedences": {
                ModelisationAllocationCP.BINARY: adding_precedence_channeling_constraint,
                ModelisationAllocationCP.CNF_COMPATIBLE: adding_precedence_channeling_constraint,
            },
        }
        for key in attributes:
            attr = getattr(additional_constraint, key)
            if attr is None:
                continue
            if self.modelisation_allocation not in functions_map[key]:
                continue
            if key == "nb_max_teams":
                constraints = functions_map[key][self.modelisation_allocation](
                    solver=self,
                    variables=self.variables["used"],
                    problem=self.problem,
                    **{key: attr},
                )
            else:
                constraints = functions_map[key][self.modelisation_allocation](
                    solver=self,
                    variables=self.variables[self.key_main_decision_variable],
                    problem=self.problem,
                    **{key: attr},
                )
            self.model += constraints

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        logger.info("Initializing model with following args:")
        for key, val in args.items():
            logger.info(f"{key} = {val}")
        modelisation_allocation = args["modelisation_allocation"]
        if modelisation_allocation == ModelisationAllocationCP.CNF_COMPATIBLE:
            raise NotImplementedError(
                "TODO: implement CNF-compatible version of allocation"
            )
        if modelisation_allocation == ModelisationAllocationCP.BINARY:
            self.modelisation_allocation = ModelisationAllocationCP.BINARY
            self.init_model_binary(**args)
            self.key_main_decision_variable = "allocation_binary"
        elif modelisation_allocation == ModelisationAllocationCP.INTEGER:
            self.modelisation_allocation = ModelisationAllocationCP.INTEGER
            self.init_model_integer(**args)
            self.key_main_decision_variable = "allocation"
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

    def init_model_binary(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        include_pair_overlap = kwargs["include_pair_overlap"]
        overlapping_advanced = kwargs["overlapping_advanced"]
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        add_lower_bound_nb_teams = kwargs["add_lower_bound_nb_teams"]
        assert include_pair_overlap or overlapping_advanced
        self.model = Model()
        domains_for_task: List[
            List[int]
        ] = self.problem.compute_allowed_team_index_all_task()
        if "allocation_binary" not in kwargs:
            allocation_binary = [
                {
                    j: boolvar(shape=1, name=f"allocation_{i}_{j}")
                    for j in domains_for_task[i]
                }
                for i in range(self.problem.number_of_activity)
            ]
        else:
            allocation_binary = kwargs["allocation_binary"]
        for i in range(len(allocation_binary)):
            # One and only one allocation
            self.model += [
                sum([allocation_binary[i][j] for j in allocation_binary[i]]) == 1
            ]
        if include_pair_overlap:
            overlaps_pair = self.problem.compute_pair_overlap_index_task()
            for i1, i2 in overlaps_pair:
                for team in allocation_binary[i1]:
                    if team in allocation_binary[i2]:
                        self.model += (
                            ~allocation_binary[i1][team] | ~allocation_binary[i2][team]
                        )
                        # cannot be both assigned
        used = boolvar(self.problem.number_of_teams, name="used")
        for i in range(len(allocation_binary)):
            for j in allocation_binary[i]:  # if a team has a task, it is used
                self.model += [allocation_binary[i][j].implies(used[j])]
        if overlapping_advanced or add_lower_bound_nb_teams:
            set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
            if add_lower_bound_nb_teams:
                max_ = max([len(o) for o in set_overlaps])
                self.model += [cp.sum(used) >= max_]
            if overlapping_advanced:
                for overlapping in set_overlaps:
                    if not (
                        (include_pair_overlap and len(overlapping) > 2)
                        or len(overlapping) >= 2
                    ):
                        # not needed to include constraint.
                        continue
                    for team in self.problem.index_to_teams_name:
                        overlap_clique: Expression = (
                            cpmpy.sum(
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
                        self.model += [overlap_clique]

        if symmbreak_on_used:  # some teams may be equivalent
            self.add_symm_breaking_constraint(model=self.model, used_variable=used)
        self.variables["obj_dict"] = {}
        if self.problem.number_of_teams >= 2:
            objectives_expr = [sum(used)]
        else:
            objectives_expr = [used]
        self.variables["obj_dict"]["nb_teams"] = objectives_expr[-1]
        keys_variable_to_log = []
        if isinstance(self.problem, TeamAllocationProblemMultiobj) and not kwargs.get(
            "ignore_dispersion", False
        ):
            for key in self.problem.attributes_cumul_activities:
                objectives_expr += [
                    self.add_multiobj(
                        key_objective=key,
                        allocation_binary=allocation_binary,
                        used=used,
                        modelisation_dispersion=kwargs["modelisation_dispersion"],
                    )
                ]
                self.variables["obj_dict"][key] = objectives_expr[-1]
                keys_variable_to_log += [key]
        self.variables["allocation_binary"] = allocation_binary
        self.variables["used"] = used
        self.model.minimize(objectives_expr[0])
        self.variables["objectives_expr"] = objectives_expr
        self.variables["keys_variable_to_log"] = keys_variable_to_log

    def init_model_integer(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        include_pair_overlap = kwargs["include_pair_overlap"]
        overlapping_advanced = kwargs["overlapping_advanced"]
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        add_lower_bound_nb_teams = kwargs["add_lower_bound_nb_teams"]
        assert include_pair_overlap or overlapping_advanced
        number_of_activities = self.problem.number_of_activity
        number_of_teams = self.problem.number_of_teams
        self.model = cpmpy.Model()
        if "allocation" in kwargs:
            allocation = kwargs["allocation"]
        else:
            allocation = cpmpy.intvar(
                lb=0,
                ub=number_of_teams - 1,
                shape=self.problem.number_of_activity,
                name="allocation",
            )
        domains_for_tasks = self.problem.compute_allowed_team_index_all_task()
        for i in range(number_of_activities):
            if len(domains_for_tasks[i]) < number_of_teams:
                self.model += [cpmpy.InDomain(allocation[i], domains_for_tasks[i])]
        used = cpmpy.boolvar(shape=number_of_teams, name="used")
        if include_pair_overlap:
            overlaps_pair = self.problem.compute_pair_overlap_index_task()
            for i1, i2 in overlaps_pair:
                self.model += [allocation[i1] != allocation[i2]]
        if overlapping_advanced or add_lower_bound_nb_teams:
            set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
            if add_lower_bound_nb_teams:
                max_ = max([len(o) for o in set_overlaps])
                self.model += [sum(used) >= max_]
            if overlapping_advanced:
                for overlapping in set_overlaps:
                    if not (
                        (include_pair_overlap and len(overlapping) > 2)
                        or len(overlapping) >= 2
                    ):
                        # not needed to include constraint.
                        continue
                    self.model += [
                        AllDifferent(
                            [
                                allocation[self.problem.index_activities_name[ac]]
                                for ac in overlapping
                            ]
                        )
                    ]
        for i in range(number_of_teams):
            self.model += [cpmpy.any(allocation == i).implies(used[i])]
        if symmbreak_on_used:
            self.add_symm_breaking_constraint(model=self.model, used_variable=used)
        # self.model.minimize(cpmpy.sum(used))
        objectives_expr = [sum(used)]
        self.variables["obj_dict"] = {"nb_teams": objectives_expr[0]}
        self.variables["allocation"] = allocation
        self.variables["used"] = used
        self.model.minimize(sum(used))
        self.variables["objectives_expr"] = objectives_expr
        self.variables["keys_variable_to_log"] = []

    def add_symm_breaking_constraint(
        self, model: cpmpy.Model, used_variable: NDVarArray
    ):
        groups = compute_equivalent_teams(team_allocation_problem=self.problem)
        for group in groups:
            for ind1, ind2 in zip(group[:-1], group[1:]):
                sym_constr = used_variable[ind2].implies(used_variable[ind1])
                # sym_constr_red = used_variable[ind1] >= used_variable[ind2]
                model += [sym_constr]
                # model += [sym_constr_red]

    def add_symm_waterfall_binary(
        self,
        model: cpmpy.Model,
        allocation_binary: List[Dict[int, cpmpy.model.Expression]],
    ):
        groups = compute_equivalent_teams(team_allocation_problem=self.problem)
        set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
        for overlaps in set_overlaps:
            overlaps = [self.problem.index_activities_name[o] for o in overlaps]
            for group in groups:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    if any(ind1 in allocation_binary[o] for o in overlaps):
                        model += cpmpy.any(
                            [
                                allocation_binary[o][ind2]
                                for o in overlaps
                                if ind2 in allocation_binary[o]
                            ]
                        ).implies(
                            cpmpy.any(
                                [
                                    allocation_binary[o][ind1]
                                    for o in overlaps
                                    if ind1 in allocation_binary[o]
                                ]
                            )
                        )

    def add_symm_waterfall_integer(
        self, model: cpmpy.Model, allocation_integer: List[cpmpy.model.Expression]
    ):
        groups = compute_equivalent_teams(team_allocation_problem=self.problem)
        set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
        for overlaps in set_overlaps:
            overlaps = [self.problem.index_activities_name[o] for o in overlaps]
            for group in groups:
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    model += (
                        cpmpy.any([allocation_integer[o] == ind2 for o in overlaps])
                    ).implies(
                        cpmpy.any([allocation_integer[o] == ind1 for o in overlaps])
                    )

    def add_multiobj(
        self,
        key_objective: str,
        allocation_binary: List[Dict[int, NDVarArray]],
        used: NDVarArray,
        modelisation_dispersion: ModelisationDispersion = ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION,
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
        dict_modeling_dispersion = model_fairness(
            used_team=used,
            allocation_variables=allocation_binary,
            value_per_task=values,
            modelisation_dispersion=modelisation_dispersion,
            number_teams=self.problem.number_of_teams,
            name_value=key_objective,
        )
        for c in dict_modeling_dispersion["constraints"]:
            self.model += c

        self.variables[f"cumulated_value_{key_objective}"] = dict_modeling_dispersion[
            "cumulated_value"
        ]
        return dict_modeling_dispersion["obj"]

    def _get_int_allocation(self):
        allocation = [None for i in range(self.problem.number_of_activity)]
        alloc_var = self.variables["allocation_binary"]
        for i in range(len(alloc_var)):
            for team in alloc_var[i]:
                if alloc_var[i][team].value():
                    allocation[i] = team
        return allocation

    def store_objective_and_time(self, verbose=False):

        self.solutions = []

        def callback():
            sol = self.retrieve_solutions(None, None)[0][0]
            eval_ = self.problem.evaluate(sol)
            objectives = {
                obj: self.variables["obj_dict"][obj].value()
                for obj in self.variables["obj_dict"]
            }
            d = eval_
            d["objs"] = objectives
            self.solutions.append((time() - self.solver_start_time, d))
            if verbose:
                print(self.solutions[-1])

        return callback

    def retrieve_solutions(
        self, result: Any, parameters_cp: ParametersCp
    ) -> ResultStorage:
        if self.modelisation_allocation in {
            ModelisationAllocationCP.BINARY,
            ModelisationAllocationCP.CNF_COMPATIBLE,
        }:
            allocation = self._get_int_allocation()
        else:
            allocation = list(self.variables["allocation"].value())
        sol = TeamAllocationSolution(problem=self.problem, allocation=allocation)
        fit = self.aggreg_from_sol(sol)
        sol._intern_objectives = {
            obj: self.variables["obj_dict"][obj].value()
            for obj in self.variables["obj_dict"]
        }
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: float = 20,
        ortools_kwargs=None,
        **args: Any,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        if self.model is None:
            self.init_model(**args)
        if self.cpm_solver is None:  # this is the first solve call
            self.cpm_solver = SolverLookup.get(self.solver_name, self.model)

        # default solver arguments for any solver
        if ortools_kwargs is not None:
            solver_kwargs = ortools_kwargs
        else:
            solver_kwargs = {}
        solver_kwargs.update(
            time_limit=time_limit, display=self.store_objective_and_time(verbose=False)
        )

        # translate parameters for specific solvers
        if self.solver_name == "ortools":
            solver_kwargs.update(
                dict(
                    num_search_workers=parameters_cp.nb_process
                    if parameters_cp.multiprocess
                    else 1,
                    log_search_progress=args.get("verbose", False),
                    display=self.store_objective_and_time(verbose=True),
                    use_lns=False,  # to make sure solutions are reproducible!
                )
            )

        if self.solver_name == "gurobi":
            solver_kwargs.update(
                dict(
                    Threads=parameters_cp.nb_process
                    if parameters_cp.multiprocess
                    else 1
                )
            )

        solver_allowed_params = inspect.signature(self.cpm_solver.solve).parameters
        if "display" not in solver_allowed_params.keys():
            solver_kwargs.pop("display")

        self.solver_start_time = time()  # used in callback to get elapsed solve-time
        res = self.cpm_solver.solve(**solver_kwargs)

        logger.info(
            f"Solver finished, found solution={res}, "
            f"objective_value= {self.cpm_solver.objective_value()} \n"
            f"status={self.cpm_solver.status()}"
        )
        self.status_solver = from_solver_status_cpmpy_to_status_solver(
            solver_status=self.cpm_solver.status()
        )
        return self.retrieve_solutions(None, parameters_cp)

    def solve_mcs(
        self,
        soft: List[cpmpy.expressions.core.Expression],
        hard: List[cpmpy.expressions.core.Expression],
        parameters_cp: Optional[ParametersCp] = None,
        time_limit=30,
        **args: Any,
    ):
        """"""
        soft = toplevel_list(soft, merge_and=False)
        assumptions = cpmpy.boolvar(shape=len(soft))
        solver: cpmpy.solvers.solver_interface.SolverInterface = cpmpy.SolverLookup.get(
            self.solver_name
        )
        solver += hard
        solver += assumptions.implies(soft)
        solver.maximize(cpmpy.sum(assumptions))
        if self.solver_name in {"exact", "pysat"}:
            res = solver.solve(time_limit=time_limit)
        else:
            solver: CPM_ortools
            res = solver.solve(
                time_limit=time_limit,
                num_search_workers=parameters_cp.nb_process
                if parameters_cp.multiprocess
                else 1,
                log_search_progress=args.get("verbose", False),
            )
        logger.info(
            f"Solver finished, found solution={res}, "
            f"objective_value= {solver.objective_value()} \n"
            f"status={solver.status()}"
        )
        dmap = dict(zip(assumptions, soft))
        return [dmap[a] for a in assumptions if a.value() is False]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def retrieve_current_solution(self) -> Solution:
        return self.retrieve_solutions(None, None)[0][0]

    def set_lexico_objective(self, obj: str) -> None:
        if self.cpm_solver is None:  # first objective, just add it to the model then
            self.model.minimize(self.variables["obj_dict"][obj])
        else:
            self.cpm_solver.minimize(self.variables["obj_dict"][obj])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        c = self.variables["obj_dict"][obj] <= value
        assert hasattr(
            self, "cpm_solver"
        ), "solver should exist! (initialized during `def solve()`)"
        self.cpm_solver += c
        # set solution hint of previous solution, if it supports it
        if "solution_hint" in self.cpm_solver.__dict__:
            self.cpm_solver.solution_hint(
                self.variables["allocation"],
                [v.value() for v in self.variables["allocation"]],
            )

        return [c]

    def get_lexico_objectives_available(self) -> list[str]:
        if self.model is not None:
            return list(self.variables["obj_dict"].keys())
        objs = ["nb_teams"]
        if isinstance(self.problem, TeamAllocationProblemMultiobj):
            for key in self.problem.attributes_cumul_activities:
                objs.append(key)
        return objs

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [sol._intern_objectives[obj] for sol, fit in res.list_solution_fits]
        return min(values)

    # def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:


class CPMpyTeamAllocationSolverStoreConstraintInfo(CPMpyTeamAllocationSolver):
    problem: TeamAllocationProblem
    hyperparameters = CPMpyTeamAllocationSolver.hyperparameters

    def __init__(
        self,
        problem: TeamAllocationProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        solver_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem: TeamAllocationProblem
        self.model: Optional[Model] = None
        self.solver_name = solver_name
        self.modelisation_allocation: Optional[ModelisationAllocationCP] = None
        self.variables = {}
        self.constraints_storage: Dict[str, Dict[Any, CONSTRAINT_ABSTRACTION]] = {}
        self.constr_to_object: Dict[Expression, CONSTRAINT_ABSTRACTION] = {}
        self.constrs_by_type: Dict[str, List[Expression]] = {}
        self.meta_constraints = []

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        if self.solver_name == "pysat":
            print(args["modelisation_allocation"])
            assert (
                args["modelisation_allocation"]
                == ModelisationAllocationCP.CNF_COMPATIBLE
            )
        self.init_model_binary(**args)
        self.modelisation_allocation = args["modelisation_allocation"]
        self.key_main_decision_variable = "allocation_binary"
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
        self.constr_to_object: Dict[Any, dataclass] = {}
        for key in self.constraints_storage:
            for sub_key in self.constraints_storage[key]:
                constr = self.constraints_storage[key][sub_key].constraint
                self.constr_to_object[constr] = self.constraints_storage[key][sub_key]
        self.constrs_by_type = {
            t: self.get_constr_of_given_type(t) for t in self.constraints_storage
        }

    def init_model_binary(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modelisation_allocation"] == ModelisationAllocationCP.CNF_COMPATIBLE:
            cnf_comp = True
        else:
            cnf_comp = False
        include_all_binary_vars = kwargs["include_all_binary_vars"]
        include_pair_overlap = kwargs["include_pair_overlap"]
        overlapping_advanced = kwargs["overlapping_advanced"]
        symmbreak_on_used = kwargs["symmbreak_on_used"]
        add_lower_bound_nb_teams = kwargs["add_lower_bound_nb_teams"]
        assert include_pair_overlap or overlapping_advanced
        self.model = Model()
        if include_all_binary_vars:
            self.constraints_storage["allowed_team"] = {}
            alloc = cpmpy.boolvar(
                (self.problem.number_of_activity, self.problem.number_of_teams),
                name="allocation",
            )
            for i in range(self.problem.number_of_activity):
                activity = self.problem.index_to_activities_name[i]
                allowed = self.problem.compute_allowed_team_index_for_task(activity)
                if len(allowed) < self.problem.number_of_teams:
                    forbidden = [
                        i
                        for i in range(self.problem.number_of_teams)
                        if i not in allowed
                    ]
                    c: Expression = ~cpmpy.any([alloc[i, k] for k in forbidden])
                    self.constraints_storage["allowed_team"][i] = AllowedTeamForTask(
                        constraint=c,
                        task=i,
                        teams=set(allowed),
                        description=f"Task {i} can only be done by teams {allowed}",
                    )
                    self.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name=f"allowed_team_{i}",
                            constraints=[c],
                            metadata={"type": "allowed_team", "task_index": i},
                        )
                    )
                    self.model += [c]
            # reorganizing the variable, to be same structure as the other option.
            allocation_binary = [
                {j: alloc[i, j] for j in range(self.problem.number_of_teams)}
                for i in range(self.problem.number_of_activity)
            ]
        else:
            domains_for_task: List[List[int]] = []
            # Take into account the allocation constraints directly in domains of variable.
            for i in range(self.problem.number_of_activity):
                activity = self.problem.index_to_activities_name[i]
                domains_for_task.append(
                    self.problem.compute_allowed_team_index_for_task(activity)
                )
            allocation_binary = [
                {
                    j: boolvar(shape=1, name=f"allocation_{i}_{j}")
                    for j in domains_for_task[i]
                }
                for i in range(self.problem.number_of_activity)
            ]
        self.constraints_storage["task_allocated"] = {}
        for i in range(len(allocation_binary)):
            # One and only one allocation
            c_i: Expression = (
                cpmpy.sum([allocation_binary[i][j] for j in allocation_binary[i]]) == 1
            )
            description_str = (
                language_dict.get("task", "task")
                + f" {i} "
                + language_dict.get("is allocated", "is allocated")
            )
            self.constraints_storage["task_allocated"][i] = AllocatedTaskConstraint(
                constraint=c_i, task=i, description=description_str
            )
            self.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"task_allocated_{i}",
                    constraints=[c_i],
                    metadata={"type": "allocated_task", "task_index": i},
                )
            )
            self.model += [c_i]
        used = boolvar(self.problem.number_of_teams, name="used")

        if include_pair_overlap:
            self.constraints_storage["pair_overlap_team"] = {}
            self.constraints_storage["pair_overlap"] = {}
            for edge in self.problem.graph_activity.edges:
                ind1 = self.problem.index_activities_name[edge[0]]
                ind2 = self.problem.index_activities_name[edge[1]]
                common_teams = [
                    team
                    for team in allocation_binary[ind1]
                    if team in allocation_binary[ind2]
                ]
                for team in common_teams:
                    overlap_pair: Expression = (
                        ~allocation_binary[ind1][team] | ~allocation_binary[ind2][team]
                    )
                    self.model += [overlap_pair]
                    if not cnf_comp:
                        overlap_pair_red: Expression = (
                            allocation_binary[ind1][team]
                            + allocation_binary[ind2][team]
                            <= 1
                        )
                        self.model += [overlap_pair_red]
                        self.constraints_storage["pair_overlap_team"][
                            (ind1, ind2, team)
                        ] = NoOverlapByTeamConstraint(
                            constraint=cpmpy.all([overlap_pair, overlap_pair_red]),
                            team=team,
                            tasks={ind1, ind2},
                            description=f"Team {team} cannot do both task {ind1} and {ind2}",
                        )
                    else:
                        self.constraints_storage["pair_overlap_team"][
                            (ind1, ind2, team)
                        ] = NoOverlapByTeamConstraint(
                            constraint=cpmpy.all([overlap_pair]),
                            team=team,
                            tasks={ind1, ind2},
                            description=f"Team {team} cannot do both task {ind1} and {ind2}",
                        )

                self.constraints_storage["pair_overlap"][
                    (ind1, ind2)
                ] = NoOverlapConstraint(
                    constraint=cpmpy.all(
                        [
                            self.constraints_storage["pair_overlap_team"][
                                (ind1, ind2, team)
                            ].constraint
                            for team in common_teams
                        ]
                    ),
                    tasks={ind1, ind2},
                    description=f"Overlapping tasks {ind1} and {ind2}",
                )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"overlapping_tasks_{ind1, ind2}",
                        constraints=[
                            self.constraints_storage["pair_overlap"][
                                (ind1, ind2)
                            ].constraint
                        ],
                    )
                )

        if overlapping_advanced or add_lower_bound_nb_teams:
            self.constraints_storage["clique_overlap"] = {}
            set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
            if add_lower_bound_nb_teams:
                max_ = max([len(o) for o in set_overlaps])
                self.model += [sum(used) >= max_]
            for overlapping in set_overlaps:
                if not (
                    (include_pair_overlap and len(overlapping) > 2)
                    or len(overlapping) >= 2
                ):
                    # not needed to include constraint.
                    continue
                no_overlap_conj = []
                for team in self.problem.index_to_teams_name:
                    overlap_clique: Expression = (
                        cpmpy.sum(
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
                    no_overlap_conj.append(overlap_clique)

                description_str = (
                    language_dict.get("overlap between tasks", "overlap between tasks")
                    + f" {sorted([self.problem.index_activities_name[x] for x in overlapping])}"
                )
                self.constraints_storage["clique_overlap"][
                    overlapping
                ] = NoOverlapConstraint(
                    constraint=cpmpy.all(no_overlap_conj),
                    tasks={self.problem.index_activities_name[x] for x in overlapping},
                    description=description_str,
                )
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name="clique_overlap",
                        constraints=[
                            self.constraints_storage["clique_overlap"][
                                overlapping
                            ].constraint
                        ],
                        metadata={
                            "type": "clique_overlap",
                            "tasks_index": {
                                self.problem.index_activities_name[x]
                                for x in overlapping
                            },
                        },
                    )
                )
                self.model += [
                    self.constraints_storage["clique_overlap"][overlapping].constraint
                ]
        self.constraints_storage["team_is_used"] = {}
        for i in range(len(allocation_binary)):
            for j in allocation_binary[i]:
                if self.problem.number_of_teams == 1:
                    if cnf_comp:
                        self.model += [allocation_binary[i][j].implies(used)]
                    else:
                        self.model += [used >= allocation_binary[i][j]]

                else:
                    if cnf_comp:
                        self.model += [allocation_binary[i][j].implies(used[j])]
                    else:
                        self.model += [used[j] >= allocation_binary[i][j]]
                if j not in self.constraints_storage["team_is_used"]:
                    self.constraints_storage["team_is_used"][j] = UsedTeamConstraint(
                        [], team=j, description=f"team {j} used"
                    )
                if self.problem.number_of_teams > 1:
                    used_j: Expression = used[j] >= allocation_binary[i][j]
                    used_j_red: Expression = allocation_binary[i][j].implies(used[j])
                else:
                    used_j: Expression = used >= allocation_binary[i][j]
                    used_j_red: Expression = allocation_binary[i][j].implies(used)
                if cnf_comp:
                    self.constraints_storage["team_is_used"][j].constraint += [
                        used_j_red
                    ]
                    self.model += [used_j_red]
                else:
                    self.constraints_storage["team_is_used"][j].constraint += [
                        used_j,
                        used_j_red,
                    ]
                    self.model += [used_j, used_j_red]
        for j in self.constraints_storage["team_is_used"]:
            self.constraints_storage["team_is_used"][j].constraint = cpmpy.all(
                self.constraints_storage["team_is_used"][j].constraint
            )
        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            self.constraints_storage["symmetries"] = {}
            for group in groups:
                symm_group = []
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    sym_constr = used[ind2].implies(used[ind1])
                    sym_constr_red = used[ind1] >= used[ind2]
                    if cnf_comp:
                        symm_group += [sym_constr]
                    else:
                        symm_group += [sym_constr, sym_constr_red]
                self.constraints_storage["symmetries"][
                    str(group)
                ] = SymmetryBreakingConstraint(
                    constraint=cpmpy.all(symm_group),
                    teams=set(group),
                    description=f"Symmetry break on teams {group}",
                )
                self.model += [
                    self.constraints_storage["symmetries"][str(group)].constraint
                ]
        if self.problem.number_of_teams > 1:
            objectives_expr = [sum(used)]
        else:
            objectives_expr = [used]
        self.variables["obj_dict"] = {"nb_teams": objectives_expr[0]}
        keys_variable_to_log = []
        if isinstance(self.problem, TeamAllocationProblemMultiobj) and not kwargs.get(
            "ignore_dispersion", False
        ):
            for key in self.problem.attributes_cumul_activities:
                objectives_expr += [
                    self.add_multiobj(
                        key_objective=key,
                        allocation_binary=allocation_binary,
                        used=used,
                    )
                ]
                self.variables["obj_dict"][key] = objectives_expr[-1]
                keys_variable_to_log += [key]
        self.variables["allocation_binary"] = allocation_binary
        self.variables["used"] = used
        if not cnf_comp:
            self.model.minimize(objectives_expr[0])
        self.variables["objectives_expr"] = objectives_expr
        self.variables["keys_variable_to_log"] = keys_variable_to_log

    def get_constr_of_given_type(self, type: str):
        if type in self.constraints_storage:
            return [
                self.constraints_storage[type][key].constraint
                for key in self.constraints_storage[type]
            ]


def adding_same_allocation_constraint_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: List[Set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["same_allocation"] = {}
    index_constr = 0
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
            constr_subset = []
            for c in common_teams:
                constr_subset.append(
                    AllEqual([variables[ind][c] for ind in index_activities])
                )
            constraints.append(cpmpy.all(constr_subset))
            if store_constraints:
                solver.constraints_storage["same_allocation"][
                    index_constr
                ] = SameAllocationConstraint(
                    constraint=constraints[-1],
                    tasks=index_activities,
                    description=f"Tasks {index_activities} allocated to same team",
                )
            index_constr += 1
    return constraints


def adding_same_allocation_constraint_cnf(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: List[Set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["same_allocation"] = {}
    index_constr = 0
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
            constr_subset = []
            for c in common_teams:
                for ind0, ind1 in zip(index_activities[:-1], index_activities[1:]):
                    constr_subset.append(variables[ind0][c] == variables[ind1][c])
                    # constraints.append(constr_subset[-1])

                # constr_subset.append(AllEqual([variables[ind][c] for ind in index_activities]))
            constraints.append(cpmpy.all(constr_subset))
            if store_constraints:
                solver.constraints_storage["same_allocation"][
                    index_constr
                ] = SameAllocationConstraint(
                    constraint=constraints[-1],
                    tasks=index_activities,
                    description=f"Tasks {index_activities} allocated to same team",
                )
            index_constr += 1

    return constraints


def adding_same_allocation_constraint_integer(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: List[Set[Hashable]],
    variables: cpmpy.model.NDVarArray,
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["same_allocation"] = {}
    for subset_activities in same_allocation:
        index_activities = [
            problem.index_activities_name[activity] for activity in subset_activities
        ]
        for i in range(len(index_activities) - 1):
            constraints.append(
                variables[index_activities[i + 1]] == variables[index_activities[i]]
            )
    return constraints


def adding_all_diff_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    all_diff_allocation: List[Set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["all_diff"] = {}
    index_constraint = 0
    for all_diff in all_diff_allocation:
        no_overlap_conj = []
        for team in problem.index_to_teams_name:
            overlap_clique: Expression = (
                cpmpy.sum(
                    [
                        variables[problem.index_activities_name[ac]][team]
                        for ac in all_diff
                        if team in variables[problem.index_activities_name[ac]]
                    ]
                )
                <= 1
            )
            no_overlap_conj.append(overlap_clique)
        constraint = cpmpy.all(no_overlap_conj)
        constraints.append(constraint)
        if store_constraints:
            description_str = (
                language_dict.get("overlap between tasks")
                + f" {[problem.index_activities_name[x] for x in all_diff]}"
            )

            solver.constraints_storage["clique_overlap"][
                index_constraint
            ] = NoOverlapConstraint(
                constraint=constraint,
                tasks={problem.index_activities_name[x] for x in all_diff},
                description=description_str,
            )
            index_constraint += 1
    return constraints


def adding_forced_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    forced_allocation: Dict[Hashable, Hashable],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["forced_alloc"] = {}
    index_constraint = 0
    for ac in forced_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_team = problem.index_teams_name[forced_allocation[ac]]
        if ind_team not in variables[ind_ac]:
            logger.warning(
                "your model is likely invalid, forced allocation not possible"
            )
        else:
            constraint_l = []
            constraint_l.append(variables[ind_ac][ind_team] == 1)
            for team in variables[ind_ac]:
                if team != ind_team:
                    constraint_l.append(variables[ind_ac][team] == 0)
            constraint = cpmpy.all(constraint_l)
            if len(constraint_l) == 1:
                constraint = constraint_l[0]
            if store_constraints:
                solver.constraints_storage["forced_alloc"][
                    index_constraint
                ] = ForcedAllocationConstraint(
                    constraint=constraint,
                    task=ind_ac,
                    team=ind_team,
                    description=f"Team {ind_team} allocated to task {ind_ac}",
                )
            index_constraint += 1
            constraints.append(constraint)
    return constraints


def adding_forbidden_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    forbidden_allocation: Dict[Hashable, Set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["forbidden_alloc"] = {}
    index_constraint = 0
    for ac in forbidden_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [
            problem.index_teams_name[t]
            for t in forbidden_allocation[ac]
            if problem.index_teams_name[t] in variables[ind_ac]
        ]
        if len(ind_teams) > 0:
            c = cpmpy.all(
                [~variables[ind_ac][i] for i in ind_teams if i in variables[ind_ac]]
            )
            constraints.append(c)
            if store_constraints:
                solver.constraints_storage["forbidden_alloc"][
                    index_constraint
                ] = ForbiddenAllocationConstraint(
                    constraint=c,
                    task=ind_ac,
                    teams=set(ind_teams),
                    description=f"Teams {set(ind_teams)} forbidden for task {ind_ac}",
                )
            index_constraint += 1
    return constraints


def adding_allowed_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    allowed_allocation: Dict[Hashable, Set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["allowed_alloc"] = {}
    index_constraint = 0
    for ac in allowed_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_teams = [problem.index_teams_name[t] for t in allowed_allocation[ac]]
        constrs = []
        for i in variables[ind_ac]:
            if i not in ind_teams:
                constrs.append(variables[ind_ac][i] == 0)
        if len(constrs) == 0:
            continue
        constr = cpmpy.all(constrs)
        if store_constraints:
            solver.constraints_storage["allowed_alloc"][
                index_constraint
            ] = AllowedTeamForTask(
                constraint=constr,
                task=ac,
                teams=allowed_allocation[ac],
                description=f"Task {ind_ac} should be done by teams {ind_teams}",
            )
        index_constraint += 1
        constraints.append(constr)
    return constraints


def adding_disjunction_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    disjunction: List[List[Tuple[Hashable, Hashable]]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> List[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["disjunction_constr"] = {}
    index_constraint = 0
    for disj in disjunction:
        c = cpmpy.any(
            [
                variables[problem.index_activities_name[x[0]]][
                    problem.index_teams_name[x[1]]
                ]
                for x in disj
            ]
        )
        if store_constraints:
            solver.constraints_storage["disjunction_constr"][
                index_constraint
            ] = DisjunctionConstraint(
                constraint=c,
                task_team_disjunction=disj,
                description=f"Disjunctive allocation" f"of {disj}",
            )
        index_constraint += 1
        constraints.append(c)
    return constraints


def adding_max_nb_teams(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    nb_max_teams: Optional[int],
    variables: List[cpmpy.model.Expression],
    problem: TeamAllocationProblem,
):
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["nb_max_teams"] = {}
    constraints = []
    if nb_max_teams is not None:
        constraints = [cpmpy.sum(variables) <= nb_max_teams]
        if store_constraints:
            solver.constraints_storage["nb_max_teams"][0] = MaxNbTeams(
                constraint=constraints[-1],
                nb_max_teams=nb_max_teams,
                description=f"No more than " f"{nb_max_teams} teams used",
            )
    return constraints


def adding_precedence_channeling_constraint(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    precedences: dict[Hashable, set[Hashable]],
    variables: List[Dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
):
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
        solver.constraints_storage["precedences"] = {}
    constraints = []
    index_constraint = 0
    if precedences is not None:
        for task in precedences:
            task_index = problem.index_activities_name[task]
            for succ in precedences[task]:
                task_succ = problem.index_activities_name[succ]
                # All equivalent
                c = cp.any(
                    [variables[task_succ][x] for x in variables[task_succ]]
                ).implies(
                    cp.any([variables[task_index][x] for x in variables[task_index]])
                )
                constraints.append(c)
                if store_constraints:
                    solver.constraints_storage["precedences"][
                        index_constraint
                    ] = TaskChanneling(
                        constraint=c,
                        tasks=(task_index, task_succ),
                        description=f"{task_succ} can only be done"
                        f"if {task_index} is also done",
                    )
    logger.info("Precedence constraints added")
    return constraints


def compute_soft_and_hard_set_of_constraint(
    cpmpy_solver: CPMpyTeamAllocationSolverStoreConstraintInfo,
    dictionnary_soft_hard: Dict[str, str] = None,
):
    soft, hard = [], []
    for key in dictionnary_soft_hard:
        if dictionnary_soft_hard[key] == "soft":
            if key in cpmpy_solver.constrs_by_type:
                soft += cpmpy_solver.constrs_by_type[key]
        elif dictionnary_soft_hard[key] == "hard":
            if key in cpmpy_solver.constrs_by_type:
                hard += cpmpy_solver.constrs_by_type[key]
    return soft, hard
