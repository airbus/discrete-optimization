#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import inspect
import json
import logging
import os
from collections import defaultdict
from collections.abc import Hashable, Iterable
from enum import Enum
from functools import reduce
from time import time
from typing import Any, Optional, Union

import cpmpy
import cpmpy as cp
from cpmpy import AllDifferent, AllEqual, Model, SolverLookup, boolvar
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
from discrete_optimization.workforce.allocation.problem import (
    AllocationAdditionalConstraint,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.solvers import TeamAllocationSolver
from discrete_optimization.workforce.allocation.utils import (
    compute_all_overlapping,
    compute_equivalent_teams,
)
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.commons.fairness_modeling_cpmpy import (
    model_fairness,
)

logger = logging.getLogger(__name__)


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


class CallbackWithBound(OrtSolutionCounter):
    def on_solution_callback(self):
        super().on_solution_callback()
        logger.debug(f"Obj bound, {self.BestObjectiveBound()}")


class CPMpyTeamAllocationSolver(CpmpySolver, TeamAllocationSolver):
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
            depends_on=(
                "modelisation_allocation",
                [
                    ModelisationAllocationCP.BINARY,
                    ModelisationAllocationCP.CNF_COMPATIBLE,
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
                    ModelisationAllocationCP.BINARY,
                    ModelisationAllocationCP.CNF_COMPATIBLE,
                ],
            ),
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
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            solver_name=solver_name,
            **kwargs,
        )
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
        domains_for_task: list[list[int]] = (
            self.problem.compute_allowed_team_index_all_task()
        )
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
        used = boolvar(shape=(self.problem.number_of_teams,), name="used")
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
                shape=(self.problem.number_of_activity,),
                name="allocation",
            )
        domains_for_tasks = self.problem.compute_allowed_team_index_all_task()
        for i in range(number_of_activities):
            if len(domains_for_tasks[i]) < number_of_teams:
                self.model += [cpmpy.InDomain(allocation[i], domains_for_tasks[i])]
        used = cpmpy.boolvar(shape=(number_of_teams,), name="used")
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
        allocation_binary: list[dict[int, cpmpy.model.Expression]],
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
        self, model: cpmpy.Model, allocation_integer: list[cpmpy.model.Expression]
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
        allocation_binary: list[dict[int, NDVarArray]],
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
                logger.debug(self.solutions[-1])

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
        assert hasattr(self, "cpm_solver"), (
            "solver should exist! (initialized during `def solve()`)"
        )
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
    def get_types_of_meta_constraints(self):
        return {mc.metadata["type"] for mc in self.meta_constraints}

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        if self.solver_name == "pysat":
            logger.debug(args["modelisation_allocation"])
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
            alloc = cpmpy.boolvar(
                shape=(self.problem.number_of_activity, self.problem.number_of_teams),
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
            domains_for_task: list[list[int]] = []
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
        for i in range(len(allocation_binary)):
            # One and only one allocation
            c_i: Expression = (
                cpmpy.sum([allocation_binary[i][j] for j in allocation_binary[i]]) == 1
            )
            self.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"task_allocated_{i}",
                    constraints=[c_i],
                    metadata={"type": "allocated_task", "task_index": i},
                )
            )
            self.model += [c_i]
        used = boolvar(shape=(self.problem.number_of_teams,), name="used")

        if include_pair_overlap:
            for edge in self.problem.graph_activity.edges:
                ind1 = self.problem.index_activities_name[edge[0]]
                ind2 = self.problem.index_activities_name[edge[1]]
                common_teams = [
                    team
                    for team in allocation_binary[ind1]
                    if team in allocation_binary[ind2]
                ]
                list_const = []
                for team in common_teams:
                    overlap_pair: Expression = (
                        ~allocation_binary[ind1][team] | ~allocation_binary[ind2][team]
                    )
                    self.model += [overlap_pair]
                    list_const.append(overlap_pair)
                    if not cnf_comp:
                        overlap_pair_red: Expression = (
                            allocation_binary[ind1][team]
                            + allocation_binary[ind2][team]
                            <= 1
                        )
                        self.model += [overlap_pair_red]
                        list_const.append(list_const)
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"overlapping_tasks_{ind1, ind2}",
                        constraints=list_const,
                    )
                )

        if overlapping_advanced or add_lower_bound_nb_teams:
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

                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name="clique_overlap",
                        constraints=[cpmpy.all(no_overlap_conj)],
                        metadata={
                            "type": "clique_overlap",
                            "tasks_index": {
                                self.problem.index_activities_name[x]
                                for x in overlapping
                            },
                        },
                    )
                )
                self.model += [cpmpy.all(no_overlap_conj)]

        dict_team_used = defaultdict(lambda: list())
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
                if self.problem.number_of_teams > 1:
                    used_j: Expression = used[j] >= allocation_binary[i][j]
                    used_j_red: Expression = allocation_binary[i][j].implies(used[j])
                else:
                    used_j: Expression = used >= allocation_binary[i][j]
                    used_j_red: Expression = allocation_binary[i][j].implies(used)
                if cnf_comp:
                    self.model += [used_j_red]
                    self.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name=f"team_{j}_is_used",
                            constraints=[used_j_red],
                            metadata={"team_index": j, "type": "team_used"},
                        )
                    )
                    dict_team_used[j].append(used_j_red)
                else:
                    self.model += [used_j, used_j_red]
                    self.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name=f"team_{j}_is_used",
                            constraints=[used_j, used_j_red],
                            metadata={"team_index": j, "type": "team_used"},
                        )
                    )
                    dict_team_used[j].extend([used_j, used_j_red])

        for j_team in dict_team_used:
            self.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"team_{j_team}_is_used",
                    constraints=dict_team_used[j_team],
                    metadata={"team_index": j_team, "type": "team_used"},
                )
            )
        if symmbreak_on_used:
            groups = compute_equivalent_teams(team_allocation_problem=self.problem)
            for group in groups:
                symm_group = []
                for ind1, ind2 in zip(group[:-1], group[1:]):
                    sym_constr = used[ind2].implies(used[ind1])
                    sym_constr_red = used[ind1] >= used[ind2]
                    if cnf_comp:
                        symm_group += [sym_constr]
                    else:
                        symm_group += [sym_constr, sym_constr_red]
                self.model += [cpmpy.all(symm_group)]
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"symmetry_{group}",
                        constraints=[cpmpy.all(symm_group)],
                        metadata={"type": "symmetry", "group": group},
                    )
                )
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


def adding_same_allocation_constraint_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: list[set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
                solver.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"Tasks {index_activities} allocated to same team",
                        constraints=[constraints[-1]],
                        metadata={"type": "same_allocation", "tasks": index_activities},
                    )
                )
            index_constr += 1
    return constraints


def adding_same_allocation_constraint_cnf(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: list[set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
                solver.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"Tasks {index_activities} allocated to same team",
                        constraints=[constraints[-1]],
                        metadata={"type": "same_allocation", "tasks": index_activities},
                    )
                )
            index_constr += 1

    return constraints


def adding_same_allocation_constraint_integer(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    same_allocation: list[set[Hashable]],
    variables: cpmpy.model.NDVarArray,
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
    for subset_activities in same_allocation:
        index_activities = [
            problem.index_activities_name[activity] for activity in subset_activities
        ]
        constr_subset = []
        for i in range(len(index_activities) - 1):
            constraints.append(
                variables[index_activities[i + 1]] == variables[index_activities[i]]
            )
            constr_subset.append(constraints[-1])
        if store_constraints:
            solver.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"Tasks {index_activities} allocated to same team",
                    constraints=constr_subset,
                    metadata={"type": "same_allocation", "tasks": index_activities},
                )
            )
    return constraints


def adding_all_diff_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    all_diff_allocation: list[set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
            solver.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"all_diff",
                    constraints=[constraint],
                    metadata={
                        "type": "all_diff",
                        "tasks": {problem.index_activities_name[x] for x in all_diff},
                    },
                )
            )
            index_constraint += 1
    return constraints


def adding_forced_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    forced_allocation: dict[Hashable, Hashable],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
    index_constraint = 0
    for ac in forced_allocation:
        ind_ac = problem.index_activities_name[ac]
        ind_team = problem.index_teams_name[forced_allocation[ac]]
        if ind_team not in variables[ind_ac]:
            logger.warning(
                "your model is likely invalid, forced allocation not possible"
            )
        else:
            constraint_l = [variables[ind_ac][ind_team] == 1]
            for team in variables[ind_ac]:
                if team != ind_team:
                    constraint_l.append(variables[ind_ac][team] == 0)
            constraint = cpmpy.all(constraint_l)
            if len(constraint_l) == 1:
                constraint = constraint_l[0]
            if store_constraints:
                solver.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"forced_alloc_{ind_ac}",
                        constraints=[constraint],
                        metadata={
                            "type": "forced_alloc",
                            "task": ind_ac,
                            "team": ind_team,
                        },
                    )
                )
            index_constraint += 1
            constraints.append(constraint)
    return constraints


def adding_forbidden_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    forbidden_allocation: dict[Hashable, set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
                solver.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name="forbidden_alloc",
                        constraints=[c],
                        metadata={
                            "type": "forbidden_alloc",
                            "task": ind_ac,
                            "teams": ind_teams,
                        },
                    )
                )
            index_constraint += 1
    return constraints


def adding_allowed_allocation_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    allowed_allocation: dict[Hashable, set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
            solver.meta_constraints.append(
                MetaCpmpyConstraint(
                    name=f"allowed_alloc",
                    constraints=[constr],
                    metadata={
                        "type": "allowed_alloc",
                        "task": ind_ac,
                        "teams": ind_teams,
                    },
                )
            )
        constraints.append(constr)
    return constraints


def adding_disjunction_binary(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    disjunction: list[list[tuple[Hashable, Hashable]]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
) -> list[cpmpy.model.Expression]:
    constraints = []
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
            solver.meta_constraints.append(
                MetaCpmpyConstraint(
                    name="disjunction_constr",
                    constraints=[c],
                    metadata={
                        "type": "disjunction",
                        "tasks": {problem.index_activities_name[x[0]] for x in disj},
                    },
                )
            )
        index_constraint += 1
        constraints.append(c)
    return constraints


def adding_max_nb_teams(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    nb_max_teams: Optional[int],
    variables: list[cpmpy.model.Expression],
    problem: TeamAllocationProblem,
):
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
    constraints = []
    if nb_max_teams is not None:
        constraints = [cpmpy.sum(variables) <= nb_max_teams]
        if store_constraints:
            solver.meta_constraints.append(
                MetaCpmpyConstraint(
                    name="nb_max_teams",
                    constraints=constraints,
                    metadata={"type": "nb_max_teams", "nb_max_teams": nb_max_teams},
                )
            )
    return constraints


def adding_precedence_channeling_constraint(
    solver: Union[
        CPMpyTeamAllocationSolver, CPMpyTeamAllocationSolverStoreConstraintInfo
    ],
    precedences: dict[Hashable, set[Hashable]],
    variables: list[dict[int, cpmpy.model.Expression]],
    problem: TeamAllocationProblem,
):
    store_constraints = False
    if isinstance(solver, CPMpyTeamAllocationSolverStoreConstraintInfo):
        store_constraints = True
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
                    solver.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name="precedences",
                            constraints=[c],
                            metadata={
                                "type": "precedences",
                                "tasks": (task_index, task_succ),
                            },
                        )
                    )
    logger.info("Precedence constraints added")
    return constraints


def compute_soft_and_hard_set_of_constraint(
    cpmpy_solver: CPMpyTeamAllocationSolverStoreConstraintInfo,
    dictionnary_soft_hard: dict[str, str] = None,
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
