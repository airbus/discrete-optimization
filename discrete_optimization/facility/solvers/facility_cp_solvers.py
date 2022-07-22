import os
import random
from dataclasses import InitVar
from datetime import timedelta
from enum import Enum

from minizinc import Instance, Model, Solver

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    compute_length_matrix,
)
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverDistanceBased,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    ParametersCP,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)


class FacilityCPModel(Enum):
    DEFAULT_INT = 0
    DEFAULT_INT_LNS = 1


file_dict = {
    FacilityCPModel.DEFAULT_INT: "facility_int.mzn",
    FacilityCPModel.DEFAULT_INT_LNS: "facility_int_lns.mzn",
}


class FacilitySolCP:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("One solution ", self.objective)
        print("Output ", _output_item)

    def check(self) -> bool:
        return True


class FacilityCP(SolverDO):
    def __init__(
        self,
        facility_problem: FacilityProblem,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        **args,
    ):
        self.facility_problem = facility_problem
        self.params_objective_function = params_objective_function
        self.cp_solver_name = cp_solver_name
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(
                self.facility_problem
            )
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.facility_problem,
            params_objective_function=params_objective_function,
        )
        self.model = None
        self.instance: Instance = None
        self.custom_output_type = False

    def init_model(self, **kwargs):
        model_type = kwargs.get("cp_model", FacilityCPModel.DEFAULT_INT)
        object_output = kwargs.get("object_output", True)
        path = os.path.join(path_minizinc, file_dict[model_type])
        self.model = Model(path)
        if object_output:
            self.model.output_type = FacilitySolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, self.model)
        instance["nb_facilities"] = self.facility_problem.facility_count
        instance["nb_customers"] = self.facility_problem.customer_count
        setup_costs, closests, distances = compute_length_matrix(self.facility_problem)
        if model_type in [FacilityCPModel.DEFAULT_INT, FacilityCPModel.DEFAULT_INT_LNS]:
            distances = [
                [int(distances[f, c]) for c in range(distances.shape[1])]
                for f in range(distances.shape[0])
            ]
            instance["distance"] = distances
            instance["setup_cost_vector"] = [int(s) for s in setup_costs]
            instance["demand"] = [
                int(self.facility_problem.customers[c].demand)
                for c in range(self.facility_problem.customer_count)
            ]
            instance["capacity"] = [
                int(self.facility_problem.facilities[f].capacity)
                for f in range(self.facility_problem.facility_count)
            ]
        else:
            distances = [
                [distances[f, c] for c in range(distances.shape[1])]
                for f in range(distances.shape[0])
            ]
            instance["distance"] = distances
            instance["setup_cost_vector"] = [s for s in setup_costs]
            instance["demand"] = [
                self.facility_problem.customers[c].demand
                for c in range(self.facility_problem.customer_count)
            ]
            instance["capacity"] = [
                self.facility_problem.facilities[f].capacity
                for f in range(self.facility_problem.facility_count)
            ]
        self.instance = instance

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        list_facility = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                if not self.custom_output_type:
                    list_facility += [result[i, "facility_for_customer"]]
                    objectives += [result[i, "objective"]]
                else:
                    list_facility += [result[i].dict["facility_for_customer"]]
                    objectives += [result[i].objective]
        else:
            if not self.custom_output_type:
                list_facility += [result["facility_for_customer"]]
                objectives += [result["objective"]]
            else:
                list_facility += [result.dict["facility_for_customer"]]
                objectives += [result.objective]
        list_solutions_fit = []
        for facility, objective in zip(list_facility, objectives):
            facility_sol = FacilitySolution(
                self.facility_problem, [f - 1 for f in facility]
            )
            fit = self.aggreg_sol(facility_sol)
            list_solutions_fit += [(facility_sol, fit)]
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(self, parameters_cp: ParametersCP = None, **kwargs) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        if self.model is None:
            self.init_model(**kwargs)
        limit_time_s = parameters_cp.TimeLimit
        result = self.instance.solve(
            timeout=timedelta(seconds=limit_time_s),
            intermediate_solutions=parameters_cp.intermediate_solution,
        )
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)

    def get_solution(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedySolverDistanceBased(self.facility_problem)
            result = greedy_solver.solve()
            solution = result.get_best_solution()
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.facility_problem.get_dummy_solution()
        print("Greedy Done")
        return solution

    def solve_lns(self, fraction_to_fix: float = 0.9, nb_iteration: int = 10, **kwargs):
        first_solution = self.get_solution(**kwargs)
        dict_color = {
            i + 1: first_solution.facility_for_customers[i] + 1
            for i in range(self.facility_problem.customer_count)
        }
        self.init_model(**kwargs)
        limit_time_s = kwargs.get("limit_time_s", 100)
        range_node = range(1, self.facility_problem.customer_count + 1)
        iteration = 0
        current_solution = first_solution
        current_best_solution = current_solution.copy()
        current_objective = self.aggreg_sol(current_best_solution)
        while iteration < nb_iteration:
            with self.instance.branch() as child:
                subpart_color = set(
                    random.sample(
                        range_node,
                        int(fraction_to_fix * self.facility_problem.customer_count),
                    )
                )
                for i in range_node:
                    if i in subpart_color:
                        child.add_string(
                            "constraint facility_for_customer["
                            + str(i)
                            + "] == "
                            + str(dict_color[i])
                            + ";\n"
                        )
                child.add_string(
                    "solve :: int_search(facility_for_customer,"
                    " input_order, indomain_min, complete) minimize(objective);\n"
                )
                print("Solving... ", iteration)
                res = child.solve(timeout=timedelta(seconds=limit_time_s))
                print(res.status)
                if res.solution is not None and -res["objective"] > current_objective:
                    current_objective = -res["objective"]
                    current_best_solution = FacilitySolution(
                        self.facility_problem,
                        [f - 1 for f in res["facility_for_customer"]],
                    )
                    self.facility_problem.evaluate(current_best_solution)
                    dict_color = {
                        i + 1: current_best_solution.facility_for_customers[i] + 1
                        for i in range(self.facility_problem.customer_count)
                    }
                    print(iteration, " : , ", res["objective"])
                    print("IMPROVED : ")
                else:
                    try:
                        print(iteration, " :  ", res["objective"])
                    except:
                        print(iteration, " failed ")
                iteration += 1
        fit = self.facility_problem.evaluate(current_best_solution)
        return current_best_solution, fit
