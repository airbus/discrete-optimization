#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from collections.abc import Iterable
from enum import Enum
from typing import Any, Union

from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.greedy import (
    GreedyFacilitySolver,
    ResultStorage,
)
from discrete_optimization.facility.solvers.lp import (
    GurobiFacilitySolver,
    MathOptFacilitySolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lns_mip import (
    GurobiConstraintHandler,
    InitialSolution,
    OrtoolsMathOptConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler

logger = logging.getLogger(__name__)


class InitialFacilityMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialFacilitySolution(InitialSolution):
    """Initial solution provider for lns algorithm.

    Attributes:
        problem (FacilityProblem): input coloring problem
        initial_method (InitialFacilityMethod): the method to use to provide the initial solution.
    """

    hyperparameters = [
        EnumHyperparameter(
            name="initial_method",
            enum=InitialFacilityMethod,
        ),
    ]

    def __init__(
        self,
        problem: FacilityProblem,
        initial_method: InitialFacilityMethod,
        params_objective_function: ParamsObjectiveFunction,
    ):
        self.problem = problem
        self.initial_method = initial_method
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=params_objective_function
        )

    def get_starting_solution(self) -> ResultStorage:
        if self.initial_method == InitialFacilityMethod.GREEDY:
            greedy_solver = GreedyFacilitySolver(
                self.problem, params_objective_function=self.params_objective_function
            )
            return greedy_solver.solve()
        else:
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg_from_sol(solution)
            return ResultStorage(
                mode_optim=self.params_objective_function.sense_function,
                list_solution_fits=[(solution, fit)],
            )


class _BaseFacilityConstraintHandler(ConstraintHandler):
    """Constraint builder used in LNS+LP for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of allocation of customer to
    facility.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def __init__(
        self,
        problem: FacilityProblem,
        fraction_to_fix: float = 0.9,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: Union[GurobiFacilitySolver, MathOptFacilitySolver],
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        subpart_customer = set(
            random.sample(
                range(self.problem.customer_count),
                int(self.fraction_to_fix * self.problem.customer_count),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, FacilitySolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a FacilitySolution."
            )
        solver.set_warm_start(current_solution)

        dict_f_fixed = {}
        for c in range(self.problem.customer_count):
            if c in subpart_customer:
                dict_f_fixed[c] = current_solution.facility_for_customers[c]
        x_var = solver.variable_decision["x"]
        lns_constraint = []
        for key in x_var:
            f, c = key
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    if not isinstance(x_var[f, c], int):
                        lns_constraint.append(
                            solver.add_linear_constraint(
                                x_var[key] == 1, name=str((f, c))
                            )
                        )
                else:
                    if not isinstance(x_var[f, c], int):
                        lns_constraint.append(
                            solver.add_linear_constraint(
                                x_var[key] == 0, name=str((f, c))
                            )
                        )
        return lns_constraint


class GurobiFacilityConstraintHandler(
    GurobiConstraintHandler, _BaseFacilityConstraintHandler
):
    """Constraint builder used in LNS+LP for coloring problem with gurobi solver.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of allocation of customer to
    facility.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.

    """

    def adding_constraint_from_results_store(
        self, solver: GurobiFacilitySolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        constraints = (
            _BaseFacilityConstraintHandler.adding_constraint_from_results_store(
                self, solver=solver, result_storage=result_storage, **kwargs
            )
        )
        solver.model.update()
        return constraints


class MathOptConstraintHandlerFacility(
    OrtoolsMathOptConstraintHandler, _BaseFacilityConstraintHandler
):
    """Constraint builder used in LNS+LP for coloring problem with mathopt solver.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of allocation of customer to
    facility.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.

    """

    def adding_constraint_from_results_store(
        self,
        solver: MathOptFacilitySolver,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        constraints = (
            _BaseFacilityConstraintHandler.adding_constraint_from_results_store(
                self, solver=solver, result_storage=result_storage, **kwargs
            )
        )
        return constraints
