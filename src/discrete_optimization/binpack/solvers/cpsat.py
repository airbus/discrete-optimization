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
    BinPackProblemBinType,
    BinPackSolution,
    Item,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
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


class CpSatBinPackBinTypeSolver(
    AllocationCpSatSolver[Item, BinPack], SchedulingCpSatSolver[Item], WarmstartMixin
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingBinPack, default=ModelingBinPack.BINARY
        )
    ]
    problem: BinPackProblemBinType
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
            args.get("upper_bound", self.problem.nb_bins), self.problem.nb_bins
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
            (i, bin_index): self.cp_model.NewBoolVar(f"alloc_{i}_{bin_index}")
            for i in range(self.problem.nb_items)
            for bin_index in range(self.upper_bound)
            if i in self.problem.list_bin_instances[bin_index].compatible_items
        }
        self.variables["allocation"] = variables_allocation
        # item in one bin
        for i in range(self.problem.nb_items):
            self.cp_model.AddExactlyOne(
                [
                    variables_allocation[(i, bin_index)]
                    for bin_index in range(self.upper_bound)
                ]
            )
        # max capacity of a bin
        for bin_index in range(self.upper_bound):
            self.cp_model.Add(
                LinearExpr.weighted_sum(
                    [
                        variables_allocation[(i, bin_index)]
                        for i in range(self.problem.nb_items)
                    ],
                    [
                        self.problem.list_items[i].weight
                        for i in range(self.problem.nb_items)
                    ],
                )
                <= self.problem.list_bin_instances[bin_index].bin_type.capacity
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
            if (task, unary_resource) in self.variables["allocation"]:
                return self.variables["allocation"][(task, unary_resource)]
            return 0
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
        capacities = [
            self.problem.list_bin_instances[i].bin_type.capacity
            for i in range(self.problem.nb_bins)
        ]
        max_capacity = max(capacities)
        min_capacity = min(capacities)
        if max_capacity == min_capacity:
            # All bins have same capacity, classical cumulative:
            self.cp_model.AddCumulative(
                [intervals[i] for i in range(self.problem.nb_items)],
                [
                    self.problem.list_items[i].weight
                    for i in range(self.problem.nb_items)
                ],
                self.problem.list_bin_instances[0].bin_type.capacity,
            )
        else:
            fake_itv = []
            fake_consumption = []
            for i in range(self.problem.nb_bins):
                if capacities[i] < max_capacity:
                    fake_itv.append(
                        self.cp_model.new_fixed_size_interval_var(
                            start=i, size=1, name=f"interval_capacity_bin_{i}"
                        )
                    )
                    fake_consumption.append(max_capacity - capacities[i])
            self.cp_model.AddCumulative(
                [intervals[i] for i in range(self.problem.nb_items)] + fake_itv,
                [
                    self.problem.list_items[i].weight
                    for i in range(self.problem.nb_items)
                ]
                + fake_consumption,
                self.problem.list_bin_instances[0].bin_type.capacity,
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
            for i, bin_index in self.variables["allocation"]:
                if solution.allocation[i] == bin_index:
                    self.cp_model.AddHint(
                        self.variables["allocation"][(i, bin_index)], 1
                    )
                else:
                    self.cp_model.AddHint(
                        self.variables["allocation"][(i, bin_index)], 0
                    )
            bins = set(solution.allocation)
            for bin_index in self.variables["used"]:
                if bin_index in bins:
                    self.cp_model.AddHint(self.variables["used"][bin_index], 1)
                else:
                    self.cp_model.AddHint(self.variables["used"][bin_index], 0)


# We implemented a more generic bin packing solver, but for retro-compatibility
CpSatBinPackSolver = CpSatBinPackBinTypeSolver
