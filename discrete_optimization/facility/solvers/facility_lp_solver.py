import random
from typing import Iterable

import mip
import numpy as np
from ortools.linear_solver import pywraplp

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import (
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    map_solver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, quicksum


def compute_length_matrix(facility_problem: FacilityProblem):
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
    facility_problem: FacilityProblem, n_cheapest: int = 10, n_shortest=10
):
    costs, closest, matrix_distance = compute_length_matrix(facility_problem)
    sorted_costs = np.argsort(costs)
    facilities = facility_problem.facilities
    customers = facility_problem.customers
    nb_facilities = len(facilities)
    nb_customers = len(customers)
    matrix_fc_indicator = np.zeros((nb_facilities, nb_customers), dtype=int)
    matrix_fc_indicator[sorted_costs[:n_cheapest], :] = 2
    for c in range(nb_customers):
        matrix_fc_indicator[closest[:n_shortest, c], c] = 2
    return matrix_fc_indicator, matrix_distance


class LP_Facility_Solver(MilpSolver):
    def __init__(
        self,
        facility_problem: FacilityProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **args
    ):
        self.facility_problem = facility_problem
        self.model = None
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.facility_problem,
            params_objective_function=params_objective_function,
        )

    def init_model(self, **kwargs):
        nb_facilities = self.facility_problem.facility_count
        nb_customers = self.facility_problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = Model("facilities")
        x = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = s.addVar(vtype=GRB.BINARY, obj=0, name="x_" + str((f, c)))
        facilities = self.facility_problem.facilities
        customers = self.facility_problem.customers
        used = s.addVars(nb_facilities, vtype=GRB.BINARY, name="y")
        constraints_customer = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.addConstr(
                quicksum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity = {}
        for f in range(nb_facilities):
            s.addConstrs(used[f] >= x[f, c] for c in range(nb_customers))
            constraint_capacity[f] = s.addConstr(
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
        self.description_constraint = {"Im lazy."}
        print("Initialized")

    def retrieve_solutions(self, range_solutions: Iterable[int]):
        solutions = []
        fits = []
        for s in range_solutions:
            solution = [0] * self.facility_problem.customer_count
            self.model.params.SolutionNumber = s
            for key in self.variable_decision["x"]:
                try:
                    value = self.variable_decision["x"][key].getAttr("Xn")
                except:
                    value = self.variable_decision["x"][
                        key
                    ]  # To deal with already fixed variable.
                if value >= 0.5:
                    f = key[0]
                    c = key[1]
                    solution[c] = f
            facility_solution = FacilitySolution(self.facility_problem, solution)
            solutions += [facility_solution]
            fits += [self.aggreg_sol(solutions[-1])]
        return ResultStorage(
            list_solution_fits=[(sol, fit) for sol, fit in zip(solutions, fits)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(self, parameters_milp: ParametersMilp = None, **kwargs):
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.model is None:
            self.init_model(**kwargs)
        limit_time_s = parameters_milp.TimeLimit
        self.model.setParam("TimeLimit", limit_time_s)
        self.model.optimize()
        nSolutions = self.model.SolCount
        nObjectives = self.model.NumObj
        objective = self.model.getObjective().getValue()
        print("Objective : ", objective)
        print("Problem has", nObjectives, "objectives")
        print("Gurobi found", nSolutions, "solutions")
        if parameters_milp.retrieve_all_solution:
            return self.retrieve_solutions(range(nSolutions))
        else:
            return self.retrieve_solutions([0])

    def fix_decision_fo_customer(self, current_solution, fraction_to_fix: float = 0.9):
        subpart_customer = set(
            random.sample(
                range(self.facility_problem.customer_count),
                int(fraction_to_fix * self.facility_problem.customer_count),
            )
        )
        dict_f_fixed = {}
        dict_f_start = {}
        for c in range(self.facility_problem.customer_count):
            dict_f_start[c] = current_solution.facility_for_customers[c]
            if c in subpart_customer:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = self.variable_decision["x"]
        lns_constraint = {}
        for key in x_var:
            f, c = key
            if f == dict_f_start[c]:
                try:
                    x_var[f, c].start = 1
                    x_var[f, c].varhintval = 1
                except:
                    pass
            else:
                try:
                    x_var[f, c].start = 0
                    x_var[f, c].varhintval = 0
                except:
                    pass
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    try:
                        lns_constraint[(f, c)] = self.model.addConstr(
                            x_var[key] == 1, name=str((f, c))
                        )
                    except:
                        print("failed")
                else:
                    try:
                        lns_constraint[(f, c)] = self.model.addConstr(
                            x_var[key] == 0, name=str((f, c))
                        )
                    except:
                        pass
        self.constraints_dict["lns"] = lns_constraint

    def fix_decision(
        self, current_solution: FacilitySolution, fraction_to_fix: float = 0.9
    ):
        self.fix_decision_fo_customer(current_solution, fraction_to_fix)

    def remove_lns_constraint(self):
        for k in self.constraints_dict["lns"]:
            self.model.remove(self.constraints_dict["lns"][k])

    def solve_lns(
        self,
        fraction_to_fix_first_iter: float = 0.0,
        fraction_to_fix: float = 0.9,
        nb_iteration: int = 10,
        **kwargs
    ):
        if self.model is None:
            print("Initialize the model")
            self.init_model(**kwargs)
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedySolverFacility(self.facility_problem)
            solution, f = greedy_solver.solve().get_best_solution_fit()
            self.start_solution = solution
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.facility_problem.get_dummy_solution()
            self.start_solution = solution
        current_solution = self.start_solution
        fit = self.facility_problem.evaluate(current_solution)
        for i in range(nb_iteration):
            print("i, ", i)
            self.fix_decision(
                current_solution,
                fraction_to_fix if i > 0 else fraction_to_fix_first_iter,
            )
            cur_sol, fit = self.solve(**kwargs).get_best_solution_fit()
            self.remove_lns_constraint()
            print(cur_sol)
            current_solution = cur_sol
        return current_solution, fit


class LP_Facility_Solver_CBC(SolverDO):
    def __init__(
        self,
        facility_problem: FacilityProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **args
    ):
        self.facility_problem = facility_problem
        self.model = None
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.facility_problem,
            params_objective_function=params_objective_function,
        )

    def init_model(self, **kwargs):
        nb_facilities = self.facility_problem.facility_count
        nb_customers = self.facility_problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = pywraplp.Solver("facility", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x = {}
        for f in range(nb_facilities):
            for c in range(nb_customers):
                if matrix_fc_indicator[f, c] == 0:
                    x[f, c] = 0
                elif matrix_fc_indicator[f, c] == 1:
                    x[f, c] = 1
                elif matrix_fc_indicator[f, c] == 2:
                    x[f, c] = s.BoolVar(name="x_" + str((f, c)))
        facilities = self.facility_problem.facilities
        customers = self.facility_problem.customers
        used = [
            s.BoolVar(name="y_" + str(j))
            for j in range(self.facility_problem.facility_count)
        ]
        constraints_customer = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.Add(
                s.Sum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity = {}
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
        self.description_constraint = {"Im lazy."}
        print("Initialized")

    def retrieve(self) -> ResultStorage:
        solution = [0] * self.facility_problem.customer_count
        for key in self.variable_decision["x"]:
            try:
                value = self.variable_decision["x"][key].solution_value()
            except:
                value = self.variable_decision["x"][
                    key
                ]  # To deal with already fixed variable.
            if value >= 0.5:
                f = key[0]
                c = key[1]
                solution[c] = f
        facility_solution = FacilitySolution(self.facility_problem, solution)
        fit = self.aggreg_sol(facility_solution)
        return ResultStorage(
            [(facility_solution, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(self, parameters_milp: ParametersMilp = None, **kwargs):
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.model is None:
            self.init_model(**kwargs)
        limit_time_s = parameters_milp.TimeLimit
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
        print("Result:", resdict[res])
        objective = self.model.Objective().Value()
        print("Objective : ", objective)
        return self.retrieve()

    def fix_decision_fo_customer(self, current_solution, fraction_to_fix: float = 0.9):
        subpart_customer = set(
            random.sample(
                range(self.facility_problem.customer_count),
                int(fraction_to_fix * self.facility_problem.customer_count),
            )
        )
        dict_f_fixed = {}
        dict_f_start = {}
        for c in range(self.facility_problem.customer_count):
            dict_f_start[c] = current_solution.facility_for_customers[c]
            if c in subpart_customer:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = self.variable_decision["x"]
        lns_constraint = {}
        for key in x_var:
            f, c = key
            if f == dict_f_start[c]:
                try:
                    self.model.SetHint([x_var[f, c]], [1])
                except:
                    pass
            else:
                try:
                    self.model.SetHint([x_var[f, c]], [0])
                except:
                    pass
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    try:
                        lns_constraint[(f, c)] = self.model.Add(x_var[key] == 1)
                    except:
                        print("failed")
                else:
                    try:
                        lns_constraint[(f, c)] = self.model.Add(x_var[key] == 0)
                    except:
                        pass
        self.constraints_dict["lns"] = lns_constraint

    def fix_decision(
        self, current_solution: FacilitySolution, fraction_to_fix: float = 0.9
    ):
        self.fix_decision_fo_customer(current_solution, fraction_to_fix)

    def remove_lns_constraint(self):
        for k in self.constraints_dict["lns"]:
            self.model.remove(self.constraints_dict["lns"][k])

    def solve_lns(
        self,
        fraction_to_fix_first_iter: float = 0.0,
        fraction_to_fix: float = 0.9,
        nb_iteration: int = 10,
        **kwargs
    ):
        if self.model is None:
            print("Initialize the model")
            self.init_model(**kwargs)
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedySolverFacility(self.facility_problem)
            solution, f = greedy_solver.solve().get_best_solution_fit()
            self.start_solution = solution
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.facility_problem.get_dummy_solution()
            self.start_solution = solution
        current_solution = self.start_solution
        fit = self.facility_problem.evaluate(current_solution)
        for i in range(nb_iteration):
            print("i, ", i)
            self.fix_decision(
                current_solution,
                fraction_to_fix if i > 0 else fraction_to_fix_first_iter,
            )
            cur_sol, fit = self.solve(**kwargs).get_best_solution_fit()
            self.remove_lns_constraint()
            print(cur_sol)
            current_solution = cur_sol
        return current_solution, fit


class LP_Facility_Solver_PyMip(LP_Facility_Solver):
    def __init__(
        self,
        facility_problem: FacilityProblem,
        milp_solver_name: MilpSolverName,
        params_objective_function: ParamsObjectiveFunction = None,
        **args
    ):
        super().__init__(
            facility_problem=facility_problem,
            params_objective_function=params_objective_function,
            **args
        )
        self.model: mip.Model = None
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]

    def init_model(self, **kwargs):
        nb_facilities = self.facility_problem.facility_count
        nb_customers = self.facility_problem.customer_count
        use_matrix_indicator_heuristic = kwargs.get(
            "use_matrix_indicator_heuristic", True
        )
        if use_matrix_indicator_heuristic:
            n_shortest = kwargs.get("n_shortest", 10)
            n_cheapest = kwargs.get("n_cheapest", 10)
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem, n_cheapest=n_cheapest, n_shortest=n_shortest
            )
        else:
            matrix_fc_indicator, matrix_length = prune_search_space(
                self.facility_problem,
                n_cheapest=nb_facilities,
                n_shortest=nb_facilities,
            )
        s = mip.Model(
            name="facilities", sense=mip.MINIMIZE, solver_name=self.solver_name
        )
        x = {}
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
        facilities = self.facility_problem.facilities
        customers = self.facility_problem.customers
        used = s.add_var_tensor((nb_facilities, 1), var_type=GRB.BINARY, name="y")
        constraints_customer = {}
        for c in range(nb_customers):
            constraints_customer[c] = s.add_constr(
                mip.xsum([x[f, c] for f in range(nb_facilities)]) == 1
            )
            # one facility
        constraint_capacity = {}
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
        self.description_constraint = {"Im lazy."}
        print("Initialized")

    def solve(self, parameters_milp: ParametersMilp, **kwargs):
        if self.model is None:
            self.init_model(**kwargs)
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.optimize(
            max_solutions=parameters_milp.n_solutions_max,
            max_seconds=parameters_milp.TimeLimit,
        )
        nSolutions = self.model.num_solutions
        objective = self.model.objective_value
        print("Objective : ", objective)
        print("Solver found", nSolutions, "solutions")
        return self.retrieve_solutions(parameters_milp)

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        solution = [0] * self.facility_problem.customer_count
        for key in self.variable_decision["x"]:
            try:
                value = self.variable_decision["x"][key].x
            except:
                value = self.variable_decision["x"][key]
            if value >= 0.5:
                f = key[0]
                c = key[1]
                solution[c] = f
        facility_solution = FacilitySolution(self.facility_problem, solution)
        result_store = ResultStorage(
            list_solution_fits=[
                (facility_solution, self.aggreg_sol(facility_solution))
            ],
            best_solution=facility_solution,
            mode_optim=self.params_objective_function.sense_function,
        )
        return result_store

    def fix_decision_fo_customer(self, current_solution, fraction_to_fix: float = 0.9):
        subpart_customer = set(
            random.sample(
                range(self.facility_problem.customer_count),
                int(fraction_to_fix * self.facility_problem.customer_count),
            )
        )
        dict_f_fixed = {}
        dict_f_start = {}
        for c in range(self.facility_problem.customer_count):
            dict_f_start[c] = current_solution.facility_for_customers[c]
            if c in subpart_customer:
                dict_f_fixed[c] = dict_f_start[c]
        x_var = self.variable_decision["x"]
        lns_constraint = {}
        for key in x_var:
            f, c = key
            if f == dict_f_start[c]:
                try:
                    x_var[f, c].start = 1
                    x_var[f, c].varhintval = 1
                except:
                    pass
            else:
                try:
                    x_var[f, c].start = 0
                    x_var[f, c].varhintval = 0
                except:
                    pass
            if c in dict_f_fixed:
                if f == dict_f_fixed[c]:
                    try:
                        lns_constraint[(f, c)] = self.model.addConstr(
                            x_var[key] == 1, name=str((f, c))
                        )
                    except:
                        print("failed")
                else:
                    try:
                        lns_constraint[(f, c)] = self.model.addConstr(
                            x_var[key] == 0, name=str((f, c))
                        )
                    except:
                        pass
        self.constraints_dict["lns"] = lns_constraint

    def fix_decision(
        self, current_solution: FacilitySolution, fraction_to_fix: float = 0.9
    ):
        self.fix_decision_fo_customer(current_solution, fraction_to_fix)

    def remove_lns_constraint(self):
        for k in self.constraints_dict["lns"]:
            self.model.remove(self.constraints_dict["lns"][k])

    def solve_lns(
        self,
        fraction_to_fix_first_iter: float = 0.0,
        fraction_to_fix: float = 0.9,
        nb_iteration: int = 10,
        **kwargs
    ):
        if self.model is None:
            print("Initialize the model")
            self.init_model(**kwargs)
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedySolverFacility(self.facility_problem)
            solution, f = greedy_solver.solve().get_best_solution_fit()
            self.start_solution = solution
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.facility_problem.get_dummy_solution()
            self.start_solution = solution
        current_solution = self.start_solution
        fit = self.facility_problem.evaluate(current_solution)
        for i in range(nb_iteration):
            print("i, ", i)
            self.fix_decision(
                current_solution,
                fraction_to_fix if i > 0 else fraction_to_fix_first_iter,
            )
            cur_sol, fit = self.solve(**kwargs).get_best_solution_fit()
            self.remove_lns_constraint()
            print(cur_sol)
            current_solution = cur_sol
        return current_solution, fit
