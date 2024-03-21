"""Linear programming models and solve functions for facility location problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import mip
import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.solvers.facility_solver import SolverFacility
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
    map_solver,
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
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.float_]]:
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
) -> Tuple[np.ndarray, np.ndarray]:
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


class _LPFacilitySolverBase(MilpSolver, SolverFacility):
    """Base class for Facility LP solvers."""

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
        self.variable_decision: Dict[str, Dict[Tuple[int, int], Union[int, Any]]] = {}
        self.constraints_dict: Dict[str, Dict[int, Any]] = {}
        self.description_variable_description = {
            "x": {
                "shape": (0, 0),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        self.description_constraint: Dict[str, Dict[str, str]] = {}

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


class LP_Facility_Solver(GurobiMilpSolver, _LPFacilitySolverBase):
    """Milp solver using gurobi library

    Attributes:
        coloring_model (FacilityProblem): facility problem instance to solve
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
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = Model("facilities")
        x: Dict[Tuple[int, int], Union[int, Any]] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = s.addVar(vtype=GRB.BINARY, obj=0, name="x_" + str((f, c)))
        facilities = self.problem.facilities
        customers = self.problem.customers
        used = s.addVars(nb_facilities, vtype=GRB.BINARY, name="y")
        constraints_customer: Dict[
            int, Union["Constr", "QConstr", "MConstr", "GenConstr"]
        ] = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.addLConstr(
                quicksum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity: Dict[
            int, Union["Constr", "QConstr", "MConstr", "GenConstr"]
        ] = {}
        for f in range(nb_facilities):
            s.addConstrs(used[f] >= x[f, c] for c in range(nb_customers))
            constraint_capacity[f] = s.addLConstr(
                quicksum([x[f, c] * customers[c].demand for c in range(nb_customers)])
                <= facilities[f].capacity
            )
        s.update()
        new_obj_f = LinExpr(0.0)
        new_obj_f += quicksum(
            [facilities[f].setup_cost * used[f] for f in range(nb_facilities)]
        )
        new_obj_f += quicksum(
            [
                matrix_length[f, c] * x[f, c]
                for f in range(nb_facilities)
                for c in range(nb_customers)
            ]
        )
        s.setObjective(new_obj_f)
        s.update()
        s.modelSense = GRB.MINIMIZE
        s.setParam(GRB.Param.Threads, 4)
        s.setParam(GRB.Param.PoolSolutions, 10000)
        s.setParam(GRB.Param.Method, 1)
        s.setParam("MIPGapAbs", 0.00001)
        s.setParam("MIPGap", 0.00000001)
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
        self.description_constraint = {"Im lazy.": {"descr": "Im lazy."}}
        logger.info("Initialized")


class LP_Facility_Solver_CBC(SolverFacility):
    """Milp formulation using cbc solver."""

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
        self.variable_decision: Dict[str, Dict[Tuple[int, int], Union[int, Any]]] = {}
        self.constraints_dict: Dict[str, Dict[int, Any]] = {}
        self.description_variable_description = {
            "x": {
                "shape": (0, 0),
                "type": bool,
                "descr": "for each facility/customer indicate"
                " if the pair is active, meaning "
                "that the customer c is dealt with facility f",
            }
        }
        self.description_constraint: Dict[str, Dict[str, str]] = {}

    def init_model(self, **kwargs: Any) -> None:
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
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
        x: Dict[Tuple[int, int], Union[int, Any]] = {}
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
        constraints_customer: Dict[int, Any] = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.Add(
                s.Sum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity: Dict[int, Any] = {}
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
        self.description_constraint = {"Im lazy.": {"descr": "Im lazy."}}
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
        return ResultStorage(
            [(facility_solution, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must be not None after calling self.init_model()."
                )
        limit_time_s = parameters_milp.time_limit
        self.model.SetTimeLimit(limit_time_s * 1000)
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


class LP_Facility_Solver_PyMip(PymipMilpSolver, _LPFacilitySolverBase):
    """Milp solver using pymip library

    Note:
        Gurobi and CBC are available backends.

    """

    def __init__(
        self,
        problem: FacilityProblem,
        milp_solver_name: MilpSolverName,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]

    def init_model(self, **kwargs: Any) -> None:
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = mip.Model(
            name="facilities", sense=mip.MINIMIZE, solver_name=self.solver_name
        )
        x: Dict[Tuple[int, int], Union[int, Any]] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = s.add_var(
                        var_type=mip.BINARY, obj=0, name="x_" + str((f, c))
                    )
        facilities = self.problem.facilities
        customers = self.problem.customers
        used = s.add_var_tensor((nb_facilities, 1), var_type=GRB.BINARY, name="y")
        constraints_customer: Dict[int, Any] = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.add_constr(
                mip.xsum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity: Dict[int, Any] = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                s.add_constr(used[f, 0] >= x[f, c])
            constraint_capacity[f] = s.add_constr(
                mip.xsum([x[f, c] * customers[c].demand for c in range(nb_customers)])
                <= facilities[f].capacity
            )
        new_obj_f = mip.LinExpr(const=0.0)
        new_obj_f.add_expr(
            mip.xsum(
                [facilities[f].setup_cost * used[f, 0] for f in range(nb_facilities)]
            )
        )
        new_obj_f.add_expr(
            mip.xsum(
                [
                    matrix_length[f, c] * x[f, c]
                    for f in range(nb_facilities)
                    for c in range(nb_customers)
                ]
            )
        )
        s.objective = new_obj_f
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
        logger.info("Initialized")
