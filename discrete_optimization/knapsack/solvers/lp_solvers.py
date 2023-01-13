#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import mip
from mip import BINARY, MAXIMIZE, xsum
from ortools.algorithms import pywrapknapsack_solver
from ortools.linear_solver import pywraplp

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    TupleFitness,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
    map_solver,
)
from discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, Constr, GenConstr, MConstr, Model, QConstr, Var, quicksum


logger = logging.getLogger(__name__)


class _BaseLPKnapsack(MilpSolver, SolverKnapsack):
    """Base class for Knapsack LP solvers."""

    def __init__(
        self,
        knapsack_model: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        self.variable_decision: Dict[str, Dict[int, Union["Var", mip.Var]]] = {}
        self.constraints_dict: Dict[
            str, Union["Constr", "QConstr", "MConstr", "GenConstr", "mip.Constr"]
        ] = {}
        self.description_variable_description: Dict[str, Dict[str, Any]] = {}
        self.description_constraint: Dict[str, Dict[str, str]] = {}
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            weight = 0.0
            xs = {}
            for (
                variable_decision_key,
                variable_decision_value,
            ) in self.variable_decision["x"].items():
                value = self.get_var_value_for_ith_solution(variable_decision_value, s)
                if value <= 0.1:
                    xs[variable_decision_key] = 0
                    continue
                xs[variable_decision_key] = 1
                weight += self.knapsack_model.index_to_item[
                    variable_decision_key
                ].weight
            solution = KnapsackSolution(
                problem=self.knapsack_model,
                value=self.get_obj_value_for_ith_solution(s),
                weight=weight,
                list_taken=[xs[e] for e in sorted(xs)],
            )
            fit = self.aggreg_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )


class LPKnapsackGurobi(GurobiMilpSolver, _BaseLPKnapsack):
    def init_model(self, **kwargs: Any) -> None:
        warm_start = kwargs.get("warm_start", {})
        self.model = Model("Knapsack")
        self.variable_decision = {"x": {}}
        self.description_variable_description = {
            "x": {
                "shape": self.knapsack_model.nb_items,
                "type": bool,
                "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not",
            }
        }
        self.description_constraint["weight"] = {
            "descr": "sum of weight of used items doesn't exceed max capacity"
        }
        weight = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        x = {}
        for item in list_item:
            i = item.index
            x[i] = self.model.addVar(
                vtype=GRB.BINARY, obj=item.value, name="x_" + str(i)
            )
            if i in warm_start:
                x[i].start = warm_start[i]
                x[i].varhintval = warm_start[i]
            weight[i] = item.weight
        self.variable_decision["x"] = x
        self.model.update()
        self.constraints_dict["weight"] = self.model.addConstr(
            quicksum([weight[i] * x[i] for i in x]) <= max_capacity
        )
        self.model.update()
        self.model.setParam("TimeLimit", 200)
        self.model.modelSense = GRB.MAXIMIZE
        self.model.setParam(GRB.Param.PoolSolutions, 10000)
        self.model.setParam("MIPGapAbs", 0.00001)
        self.model.setParam("MIPGap", 0.00000001)

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        # We call explicitely the method to be sure getting the proper one
        return _BaseLPKnapsack.retrieve_solutions(self, parameters_milp=parameters_milp)


class LPKnapsackCBC(SolverKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        self.model: Optional[pywraplp.Solver] = None
        self.variable_decision: Dict[str, Dict[int, Any]] = {}
        self.constraints_dict: Dict[str, Any] = {}
        self.description_variable_description: Dict[str, Dict[str, Any]] = {}
        self.description_constraint: Dict[str, Dict[str, str]] = {}
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )

    def init_model(
        self, warm_start: Optional[Dict[int, int]] = None, **kwargs: Any
    ) -> None:
        if warm_start is None:
            warm_start = {}
        self.description_variable_description = {
            "x": {
                "shape": self.knapsack_model.nb_items,
                "type": bool,
                "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not",
            }
        }
        self.description_constraint["weight"] = {
            "descr": "sum of weight of used items doesn't exceed max capacity"
        }
        S = pywraplp.Solver("knapsack", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x = {}
        weight = {}
        value = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        for item in list_item:
            i = item.index
            x[i] = S.BoolVar("x_" + str(i))
            if i in warm_start:
                S.SetHint([x[i]], [warm_start[i]])
            weight[i] = item.weight
            value[i] = item.value
        self.constraints_dict["weight"] = S.Add(
            S.Sum([x[i] * weight[i] for i in x]) <= max_capacity
        )
        value_knap = S.Sum([x[i] * value[i] for i in x])
        S.Maximize(value_knap)
        self.model = S
        self.variable_decision["x"] = x

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.model is None:
            self.init_model()
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.model.SetTimeLimit(60000)
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
        logger.debug(f"Result: {resdict[res]}")
        objective = self.model.Objective().Value()
        xs = {}
        x = self.variable_decision["x"]
        weight = 0.0
        for i in x:
            sv = x[i].solution_value()
            if sv >= 0.5:
                xs[i] = 1
                weight += self.knapsack_model.index_to_item[i].weight
            else:
                xs[i] = 0
        sol = KnapsackSolution(
            problem=self.knapsack_model,
            value=objective,
            weight=weight,
            list_taken=[xs[e] for e in sorted(xs)],
        )
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )


# Can use GRB or CBC
class LPKnapsack(PymipMilpSolver, _BaseLPKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        milp_solver_name: MilpSolverName,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            knapsack_model=knapsack_model,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]

    def init_model(self, **kwargs: Any) -> None:
        warm_start = kwargs.get("warm_start", {})
        solver_name = kwargs.get("solver_name", self.solver_name)
        self.model = MyModelMilp("Knapsack", solver_name=solver_name, sense=MAXIMIZE)
        self.variable_decision = {"x": {}}
        self.description_variable_description = {
            "x": {
                "shape": self.knapsack_model.nb_items,
                "type": bool,
                "descr": "dictionary with key the item index \
                                                                 and value the boolean value corresponding \
                                                                 to taking the item or not",
            }
        }
        self.description_constraint["weight"] = {
            "descr": "sum of weight of used items doesn't exceed max capacity"
        }
        weight = {}
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        x = {}
        start = []
        for item in list_item:
            i = item.index
            x[i] = self.model.add_var(
                var_type=BINARY, obj=item.value, name="x_" + str(i)
            )
            if i in warm_start:
                start += [(x[i], warm_start[i])]
            weight[i] = item.weight
        self.model.start = start
        self.variable_decision["x"] = x
        self.model.update()
        self.constraints_dict["weight"] = self.model.add_constr(
            xsum([weight[i] * x[i] for i in x]) <= max_capacity, name="capacity_constr"
        )
        self.model.update()

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        # We call explicitely the method to be sure getting the proper one
        return _BaseLPKnapsack.retrieve_solutions(self, parameters_milp=parameters_milp)


class KnapsackORTools(SolverKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        self.model = None
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )
        list_item = self.knapsack_model.list_items
        max_capacity = self.knapsack_model.max_capacity
        values = [item.value for item in list_item]
        weights = [[item.weight for item in list_item]]
        capacities = [max_capacity]
        solver.Init(values, weights, capacities)
        self.model = solver

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        computed_value = self.model.Solve()
        logger.debug(f"Total value = {computed_value}")
        xs = {}
        weight = 0
        value = 0
        for i in range(self.knapsack_model.nb_items):
            if self.model.BestSolutionContains(i):
                weight += self.knapsack_model.list_items[i].weight
                value += self.knapsack_model.list_items[i].value
                xs[self.knapsack_model.list_items[i].index] = 1
            else:
                xs[self.knapsack_model.list_items[i].index] = 0
        sol = KnapsackSolution(
            problem=self.knapsack_model,
            value=value,
            weight=weight,
            list_taken=[xs[e] for e in sorted(xs)],
        )
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
