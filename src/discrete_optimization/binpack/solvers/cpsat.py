#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections.abc import Hashable
from enum import Enum
from typing import Any, Iterable, Optional, Union

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    IntVar,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.binpack.problem import (
    BinPack,
    BinPackProblem,
    BinPackSolution,
    Item,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)

logger = logging.getLogger(__name__)


class ModelingBinPack(Enum):
    BINARY = 0
    SCHEDULING = 1


class ModelingError(Exception):
    """Exception raised when calling for variables not existing for chosen modeling."""

    ...


class CpSatBinPackSolver(
    AllocationCpSatSolver[Item, BinPack], SchedulingCpSatSolver[Item], WarmstartMixin
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingBinPack, default=ModelingBinPack.BINARY
        )
    ]
    problem: BinPackProblem
    modeling: ModelingBinPack

    def __init__(
        self,
        problem: BinPackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables: dict[str, Union[IntVar, dict[Hashable, IntVar]]] = {}
        self.upper_bound = self.problem.nb_items  # upper bound on nb of bin packs

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"Obj={cpsolvercb.objective_value}, Bound={cpsolvercb.best_objective_bound}"
        )
        allocation: list[Optional[int]] = [None for i in range(self.problem.nb_items)]
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
        # store modeling and upper_bound (=> allowed resources)
        self.modeling = args["modeling"]
        self.upper_bound = min(
            args.get("upper_bound", self.problem.nb_items), self.problem.nb_items
        )
        if args["modeling"] == ModelingBinPack.BINARY:
            self.init_model_binary(**args)
        elif args["modeling"] == ModelingBinPack.SCHEDULING:
            self.init_model_scheduling(**args)

    @property
    def subset_unaryresources_allowed(self) -> Iterable[BinPack]:
        return range(self.upper_bound)

    def init_model_binary(self, **args: Any):
        super().init_model(**args)

        # boolean allocation variables
        variables_allocation = {
            (i, bin_): self.cp_model.NewBoolVar(f"alloc_{i}_{bin_}")
            for i in range(self.problem.nb_items)
            for bin_ in range(self.upper_bound)
        }
        self.variables["allocation"] = variables_allocation
        # item in one bin
        for i in range(self.problem.nb_items):
            self.cp_model.AddExactlyOne(
                [variables_allocation[(i, bin)] for bin in range(self.upper_bound)]
            )
        # max capacity of a bin
        for bin_ in range(self.upper_bound):
            self.cp_model.Add(
                LinearExpr.weighted_sum(
                    [
                        variables_allocation[(i, bin_)]
                        for i in range(self.problem.nb_items)
                    ],
                    [
                        self.problem.list_items[i].weight
                        for i in range(self.problem.nb_items)
                    ],
                )
                <= self.problem.capacity_bin
            )
        # bin used variable
        self.create_used_variables()
        self.variables["used"] = self.used_variables
        # constraints on bin ordering
        for bin_ in range(self.upper_bound):
            if bin_ >= 1:
                self.cp_model.add(
                    self.used_variables[bin_] <= self.used_variables[bin_ - 1]
                )
        # incompatible items
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                for bin_ in range(self.upper_bound):
                    self.cp_model.AddForbiddenAssignments(
                        [
                            variables_allocation[(i, bin_)],
                            variables_allocation[(j, bin_)],
                        ],
                        [(1, 1)],
                    )

        self.cp_model.Minimize(self.get_nb_unary_resources_used_variable())

    def get_task_unary_resource_is_present_variable(
        self, task: Item, unary_resource: BinPack
    ) -> LinearExprT:
        if self.modeling == ModelingBinPack.BINARY:
            return self.variables["allocation"][(task, unary_resource)]
        else:
            raise ModelingError(f"No allocation variable with {self.modeling}")

    def init_model_scheduling(self, **args: Any):
        super().init_model(**args)
        starts = {}
        intervals = {}
        for i in range(self.problem.nb_items):
            starts[i] = self.cp_model.NewIntVar(
                lb=0, ub=self.upper_bound, name=f"bin_{i}"
            )
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
        makespan = self.get_global_makespan_variable()
        self.variables["makespan"] = makespan
        self.cp_model.minimize(makespan)

    def get_makespan_upper_bound(self) -> int:
        return self.upper_bound + 1

    def get_task_start_or_end_variable(
        self, task: Item, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if not self.modeling == ModelingBinPack.SCHEDULING:
            raise ModelingError(f"No start or end variable with {self.modeling}")
        var = self.variables["starts"][task]
        if start_or_end == StartOrEnd.END:
            var = var + 1
        return var

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
            for i, bin_ in self.variables["allocation"]:
                if solution.allocation[i] == bin_:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin_)], 1)
                else:
                    self.cp_model.AddHint(self.variables["allocation"][(i, bin_)], 0)
            bins = set(solution.allocation)
            for bin_ in self.variables["used"]:
                if bin_ in bins:
                    self.cp_model.AddHint(self.variables["used"][bin_], 1)
                else:
                    self.cp_model.AddHint(self.variables["used"][bin_], 0)
