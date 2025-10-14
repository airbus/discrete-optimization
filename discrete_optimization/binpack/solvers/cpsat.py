#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math
from enum import Enum
from typing import Any, Optional, Union

from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar, LinearExpr

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class ModelingBinPack(Enum):
    BINARY = 0
    SCHEDULING = 1


class CpSatBinPackSolver(OrtoolsCpSatSolver, WarmstartMixin):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingBinPack, default=ModelingBinPack.BINARY
        )
    ]
    problem: BinPackProblem
    modeling: ModelingBinPack

    def __init__(
        self,
        problem: BinPackSolution,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables: dict[
            str, Union[IntVar, list[IntVar], list[dict[int, IntVar]]]
        ] = {}

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        allocation = [None for i in range(self.problem.nb_items)]
        if self.modeling == ModelingBinPack.BINARY:
            for i, j in self.variables["allocation"]:
                if cpsolvercb.Value(self.variables["allocation"][(i, j)]) == 1:
                    allocation[i] = j
        if self.modeling == ModelingBinPack.SCHEDULING:
            for i in self.variables["starts"]:
                allocation[i] = cpsolvercb.Value(self.variables["starts"][i])
        return BinPackSolution(problem=self.problem, allocation=allocation)

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        if args["modeling"] == ModelingBinPack.BINARY:
            self.init_model_binary(**args)
            self.modeling = args["modeling"]
        if args["modeling"] == ModelingBinPack.SCHEDULING:
            self.init_model_scheduling(**args)
            self.modeling = args["modeling"]

    def init_model_binary(self, **args: Any):
        super().init_model(**args)
        variables_allocation = {}
        used_bin = {}
        upper_bound = args.get("upper_bound", self.problem.nb_items)
        for bin in range(upper_bound):
            used_bin[bin] = self.cp_model.NewBoolVar(f"used_{bin}")
            if bin >= 1:
                self.cp_model.Add(used_bin[bin] <= used_bin[bin - 1])
        for i in range(self.problem.nb_items):
            for bin in range(upper_bound):
                variables_allocation[(i, bin)] = self.cp_model.NewBoolVar(
                    f"alloc_{i}_{bin}"
                )
                self.cp_model.Add(used_bin[bin] >= variables_allocation[(i, bin)])
            self.cp_model.AddExactlyOne(
                [variables_allocation[(i, bin)] for bin in range(upper_bound)]
            )
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                for bin in range(upper_bound):
                    self.cp_model.AddForbiddenAssignments(
                        [
                            variables_allocation[(i, bin)],
                            variables_allocation[(j, bin)],
                        ],
                        [(1, 1)],
                    )
        for bin in range(upper_bound):
            self.cp_model.Add(
                LinearExpr.weighted_sum(
                    [
                        variables_allocation[(i, bin)]
                        for i in range(self.problem.nb_items)
                    ],
                    [
                        self.problem.list_items[i].weight
                        for i in range(self.problem.nb_items)
                    ],
                )
                <= self.problem.capacity_bin
            )
        self.variables["allocation"] = variables_allocation
        self.variables["used"] = used_bin
        self.cp_model.Minimize(sum([used_bin[bin] for bin in used_bin]))

    def init_model_scheduling(self, **args: Any):
        super().init_model(**args)
        weights = [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        upper_bound = args.get("upper_bound", self.problem.nb_items)
        # nb_min_bins = int(math.ceil(sum(weights) / float(self.problem.capacity_bin)))
        # nb_max_bins = min(self.problem.nb_items, 2 * nb_min_bins)
        # upper_bound = min(upper_bound, nb_max_bins)
        starts = {}
        intervals = {}
        for i in range(self.problem.nb_items):
            starts[i] = self.cp_model.NewIntVar(lb=0, ub=upper_bound, name=f"bin_{i}")
            intervals[i] = self.cp_model.NewFixedSizeIntervalVar(
                start=starts[i], size=1, name=f"interval_{i}"
            )
        self.cp_model.AddCumulative(
            [intervals[i] for i in range(self.problem.nb_items)],
            [self.problem.list_items[i].weight for i in range(self.problem.nb_items)],
            self.problem.capacity_bin,
        )
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                self.cp_model.Add(starts[i] != starts[j])
        self.variables["starts"] = starts
        makespan = self.cp_model.NewIntVar(lb=1, ub=upper_bound + 1, name="nb_bins")
        self.variables["makespan"] = makespan
        self.cp_model.add_max_equality(makespan, [starts[i] + 1 for i in starts])
        self.cp_model.minimize(makespan)

    def set_warm_start(self, solution: BinPackSolution) -> None:
        if self.modeling == ModelingBinPack.SCHEDULING:
            self.cp_model.ClearHints()
            for i in range(self.problem.nb_items):
                self.cp_model.AddHint(
                    self.variables["starts"][i], solution.allocation[i]
                )
            self.cp_model.AddHint(
                self.variables["makespan"], max(solution.allocation) + 1
            )
        if self.modeling == ModelingBinPack.BINARY:
            self.cp_model.ClearHints()
            for i, bin in self.variables["allocation"]:
                if solution.allocation[i] == bin:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin)], 1)
                else:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin)], 0)
            bins = set(solution.allocation)
            for bin in self.variables["used"]:
                if bin in bins:
                    self.cp_model.AddHint(self.variables["used"][bin], 1)
                else:
                    self.cp_model.AddHint(self.variables["used"][bin], 0)
