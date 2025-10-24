"""Linear programming models and solve functions for facility location problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp
from ortools.math_opt.python import mathopt

from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers import FacilitySolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    ConstraintType,
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import (
        GRB,
        Constr,
        GenConstr,
        LinExpr,
        MConstr,
        Model,
        QConstr,
        quicksum,
    )


def compute_length_matrix(
    facility_problem: FacilityProblem,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Precompute all the cost of allocation in a matrix form.

    A matrix "closest" is also computed, sorting for each customers the facility by distance.

    Args:
        facility_problem (FacilityProblem): facility problem instance to compute cost matrix

    Returns: setup cost vector, sorted matrix distance, matrix distance

    """
    facilities = facility_problem.facilities
    customers = facility_problem.customers
    nb_facilities = len(facilities)
    nb_customers = len(customers)
    matrix_distance = np.zeros((nb_facilities, nb_customers))
    costs = np.array([facilities[f].setup_cost for f in range(nb_facilities)])
    for f in range(nb_facilities):
        for c in range(nb_customers):
            matrix_distance[f, c] = facility_problem.evaluate_customer_facility(
                facilities[f], customers[c]
            )
    closest = np.argsort(matrix_distance, axis=0)
    return costs, closest, matrix_distance


def prune_search_space(
    facility_problem: FacilityProblem, n_cheapest: int = 10, n_shortest: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Utility function that can prune the search space.

    Output of this function will be used to :
    - consider only the n_cheapest facility that has the cheapest setup_cost
    - consider only the n_shortest (closest actually) facilities for each customers


    Args:
        facility_problem (FacilityProblem): facility problem instance
        n_cheapest (int): select the cheapest setup cost facilities
        n_shortest (int): for each customer, select the closest facilities

    Returns: tuple of matrix, first element is a matrix (facility_count, customer_count) with 2 as value
    when we should consider the allocation possible. Second element in the (facility,customer) matrix distance.
    """
    costs, closest, matrix_distance = compute_length_matrix(facility_problem)
    sorted_costs = np.argsort(costs)
    facilities = facility_problem.facilities
    customers = facility_problem.customers
    nb_facilities = len(facilities)
    nb_customers = len(customers)
    matrix_fc_indicator = np.zeros((nb_facilities, nb_customers), dtype=np.int_)
    matrix_fc_indicator[sorted_costs[:n_cheapest], :] = 2
    for c in range(nb_customers):
        matrix_fc_indicator[closest[:n_shortest, c], c] = 2
    return matrix_fc_indicator, matrix_distance


class _BaseLpFacilitySolver(MilpSolver, FacilitySolver):
    """Base class for Facility LP solvers."""

    hyperparameters = [
        CategoricalHyperparameter(
            name="use_matrix_indicator_heuristic", default=True, choices=[False, True]
        ),
        IntegerHyperparameter(
            name="n_shortest",
            default=10,
            low=0,
            high=100,
            depends_on=("use_matrix_indicator_heuristic", [True]),
        ),
        IntegerHyperparameter(
            name="n_cheapest",
            default=10,
            low=0,
            high=100,
            depends_on=("use_matrix_indicator_heuristic", [True]),
        ),
    ]

    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.model = None
        self.variable_decision: dict[
            str, dict[Union[int, tuple[int, int]], Union[int, Any]]
        ] = {}
        self.constraints_dict: dict[str, dict[int, Any]] = {}
        self.description_variable_description = {
            "x": {
                "shape": (0, 0),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        self.description_constraint: dict[str, dict[str, str]] = {}

    def init_model(self, **kwargs: Any) -> None:
        """

        Keyword Args:
            use_matrix_indicator_heuristic (bool): use the prune search method to reduce number of variable.
            n_shortest (int): parameter for the prune search method
            n_cheapest (int): parameter for the prune search method

        Returns: None

        """
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        use_matrix_indicator_heuristic = kwargs["use_matrix_indicator_heuristic"]
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs["n_shortest"]
            n_cheapest = kwargs["n_cheapest"]
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        self.model = self.create_empty_model(name="facilities")
        x: dict[tuple[int, int], Union[int, Any]] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = self.add_binary_variable(name="x_" + str((f, c)))
        facilities = self.problem.facilities
        customers = self.problem.customers
        used = [self.add_binary_variable(name=f"y_{i}") for i in range(nb_facilities)]
        constraints_customer: dict[int, ConstraintType] = {}
        for c in range(nb_customers):
            constraints_customer[c] = self.add_linear_constraint(
                self.construct_linear_sum(x[f, c] for f in range(nb_facilities)) == 1
            )
            # one facility
        constraint_capacity: dict[int, ConstraintType] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                self.add_linear_constraint(used[f] >= x[f, c])
            constraint_capacity[f] = self.add_linear_constraint(
                self.construct_linear_sum(
                    x[f, c] * customers[c].demand for c in range(nb_customers)
                )
                <= facilities[f].capacity
            )
        new_obj_f = self.construct_linear_sum(
            facilities[f].setup_cost * used[f] for f in range(nb_facilities)
        )
        new_obj_f += self.construct_linear_sum(
            matrix_length[f, c] * x[f, c]
            for f in range(nb_facilities)
            for c in range(nb_customers)
        )
        self.set_model_objective(new_obj_f, minimize=True)
        self.variable_decision = {"x": x, "y": used}
        self.constraints_dict = {
            "constraint_customer": constraints_customer,
            "constraint_capacity": constraint_capacity,
        }
        self.description_variable_description = {
            "x": {
                "shape": (nb_facilities, nb_customers),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        logger.info("Initialized")

    def convert_to_variable_values(
        self, solution: FacilitySolution
    ) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        # Init all variables to 0
        hinted_variables = {var: 0 for var in self.variable_decision["x"].values()}
        # Set var(facility, customer) to 1 according to the solution
        for c, f in enumerate(solution.facility_for_customers):
            variable_decision_key = (f, c)
            hinted_variables[self.variable_decision["x"][variable_decision_key]] = 1
            hinted_variables[self.variable_decision["y"][f]] = 1

        return hinted_variables

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> FacilitySolution:
        facility_for_customer = [0] * self.problem.customer_count
        for (
            variable_decision_key,
            variable_decision_value,
        ) in self.variable_decision["x"].items():
            if not isinstance(variable_decision_value, int):
                value = get_var_value_for_current_solution(variable_decision_value)
            else:
                value = variable_decision_value
            if value >= 0.5:
                f = variable_decision_key[0]
                c = variable_decision_key[1]
                facility_for_customer[c] = f
        return FacilitySolution(self.problem, facility_for_customer)


class GurobiFacilitySolver(GurobiMilpSolver, _BaseLpFacilitySolver):
    """Milp solver using gurobi library

    Attributes:
        coloring_problem (FacilityProblem): facility problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                        (however this is just used for the ResultStorage creation, not in the optimisation)
    """

    def init_model(self, **kwargs: Any) -> None:
        """

        Keyword Args:
            use_matrix_indicator_heuristic (bool): use the prune search method to reduce number of variable.
            n_shortest (int): parameter for the prune search method
            n_cheapest (int): parameter for the prune search method

        Returns: None

        """
        _BaseLpFacilitySolver.init_model(self, **kwargs)
        self.model.setParam(GRB.Param.Threads, 4)
        self.model.setParam(GRB.Param.Method, 1)
        self.model.update()

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return _BaseLpFacilitySolver.convert_to_variable_values(self, solution)


class MathOptFacilitySolver(OrtoolsMathOptMilpSolver, _BaseLpFacilitySolver):
    """Milp solver using gurobi library

    Attributes:
        coloring_problem (FacilityProblem): facility problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                        (however this is just used for the ResultStorage creation, not in the optimisation)
    """

    hyperparameters = _BaseLpFacilitySolver.hyperparameters

    def convert_to_variable_values(
        self, solution: FacilitySolution
    ) -> dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        See https://or-tools.github.io/docs/pdoc/ortools/math_opt/python/model_parameters.html#SolutionHint
        for more information.

        """
        return _BaseLpFacilitySolver.convert_to_variable_values(self, solution)


class CbcFacilitySolver(FacilitySolver):
    """Milp formulation using cbc solver."""

    hyperparameters = [
        CategoricalHyperparameter(
            name="use_matrix_indicator_heuristic", default=True, choices=[False, True]
        ),
        IntegerHyperparameter(name="n_shortest", default=10, low=0, high=100),
        IntegerHyperparameter(name="n_cheapest", default=10, low=0, high=100),
    ]

    model: Optional[pywraplp.Solver]

    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.model = None
        self.variable_decision: dict[str, dict[tuple[int, int], Union[int, Any]]] = {}
        self.constraints_dict: dict[str, dict[int, Any]] = {}
        self.description_variable_description = {
            "x": {
                "shape": (0, 0),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        self.description_constraint: dict[str, dict[str, str]] = {}

    def init_model(self, **kwargs: Any) -> None:
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        use_matrix_indicator_heuristic = kwargs["use_matrix_indicator_heuristic"]
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs["n_shortest"]
            n_cheapest = kwargs["n_cheapest"]
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = pywraplp.Solver("facility", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x: dict[tuple[int, int], Union[int, Any]] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = s.BoolVar(name="x_" + str((f, c)))
        facilities = self.problem.facilities
        customers = self.problem.customers
        used = [
            s.BoolVar(name="y_" + str(j)) for j in range(self.problem.facility_count)
        ]
        constraints_customer: dict[int, Any] = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.Add(
                s.Sum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity: dict[int, Any] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                s.Add(used[f] >= x[f, c])
            constraint_capacity[f] = s.Add(
                s.Sum([x[f, c] * customers[c].demand for c in range(nb_customers)])
                <= facilities[f].capacity
            )
        obj = s.Sum(
            [facilities[f].setup_cost * used[f] for f in range(nb_facilities)]
            + [
                matrix_length[f, c] * x[f, c]
                for f in range(nb_facilities)
                for c in range(nb_customers)
            ]
        )
        s.Minimize(obj)
        self.model = s
        self.variable_decision = {"x": x}
        self.constraints_dict = {
            "constraint_customer": constraints_customer,
            "constraint_capacity": constraint_capacity,
        }
        self.description_variable_description = {
            "x": {
                "shape": (nb_facilities, nb_customers),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        self.description_constraint = {}
        logger.info("Initialized")

    def retrieve(self) -> ResultStorage:
        solution = [0] * self.problem.customer_count
        for key in self.variable_decision["x"]:
            variable_decision_key = self.variable_decision["x"][key]
            if not isinstance(variable_decision_key, int):
                value = variable_decision_key.solution_value()
            else:
                value = variable_decision_key
            if value >= 0.5:
                f = key[0]
                c = key[1]
                solution[c] = f
        facility_solution = FacilitySolution(self.problem, solution)
        fit = self.aggreg_from_sol(facility_solution)
        return self.create_result_storage(
            [(facility_solution, fit)],
        )

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[int] = 30,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must be not None after calling self.init_model()."
                )
        if time_limit is not None:
            self.model.SetTimeLimit(time_limit * 1000)
        res = self.model.Solve()
        resdict = {
            0: "OPTIMAL",
            1: "FEASIBLE",
            2: "INFEASIBLE",
            3: "UNBOUNDED",
            4: "ABNORMAL",
            5: "MODEL_INVALID",
            6: "NOT_SOLVED",
        }
        logger.info(f"Result: {resdict[res]}")
        objective = self.model.Objective().Value()
        logger.info(f"Objective : {objective}")
        return self.retrieve()
