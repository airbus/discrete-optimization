import os
import random
from dataclasses import InitVar
from datetime import timedelta
from enum import Enum

import networkx as nx
import pymzn
from minizinc import Instance, Model, Solver

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_toolbox import compute_cliques
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    CPSolverName,
    ParametersCP,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage

path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)


class ColotingCPSolution:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("New solution ", self.objective)
        print("Output ", _output_item)

    def check(self) -> bool:
        return True


class ColoringCPModel(Enum):
    CLIQUES = 0
    DEFAULT = 1
    LNS = 2


file_dict = {
    ColoringCPModel.CLIQUES: "coloring_clique.mzn",
    ColoringCPModel.DEFAULT: "coloring.mzn",
    ColoringCPModel.LNS: "coloring_for_lns.mzn",
}


class ColoringCP(CPSolver):
    def __init__(
        self,
        coloring_problem: ColoringProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        **args,
    ):
        self.coloring_problem = coloring_problem
        self.number_of_nodes = self.coloring_problem.number_of_nodes
        self.number_of_edges = len(self.coloring_problem.graph.edges_infos_dict)
        self.nodes_name = self.coloring_problem.graph.nodes_name
        self.index_nodes_name = self.coloring_problem.index_nodes_name
        self.index_to_nodes_name = self.coloring_problem.index_to_nodes_name
        self.graph = self.coloring_problem.graph
        self.model: Model = None
        self.instance: Instance = None
        self.custom_output_type = False
        self.g = None
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.coloring_problem,
            params_objective_function=params_objective_function,
        )
        self.cp_solver_name = cp_solver_name

    def init_model(self, **kwargs):
        nb_colors = kwargs.get("nb_colors", None)
        object_output = kwargs.get("object_output", True)
        include_seq_chain_constraint = kwargs.get("include_seq_chain_constraint", False)
        if nb_colors is None:
            solution = self.get_solution(**kwargs)
            nb_colors = solution.nb_color
        model_type = kwargs.get("cp_model", ColoringCPModel.DEFAULT)
        path = os.path.join(path_minizinc, file_dict[model_type])
        self.model = Model(path)
        if object_output:
            self.model.output_type = ColotingCPSolution
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, self.model)
        instance["n_nodes"] = self.number_of_nodes
        instance["n_edges"] = int(self.number_of_edges / 2)
        instance["nb_colors"] = nb_colors
        keys = []
        if model_type in {ColoringCPModel.DEFAULT, ColoringCPModel.CLIQUES}:
            instance["include_seq_chain_constraint"] = include_seq_chain_constraint
            keys += ["include_seq_chain_constraint"]
        print("nb colors ", nb_colors)
        keys += ["n_nodes", "n_edges", "nb_colors"]
        edges = [
            [self.index_nodes_name[e[0]] + 1, self.index_nodes_name[e[1]] + 1, e[2]]
            for e in self.coloring_problem.graph.edges
        ]
        g = nx.Graph()
        g.add_nodes_from([i for i in range(1, self.number_of_nodes + 1)])
        g.add_edges_from(edges)
        self.g = g
        if model_type == ColoringCPModel.CLIQUES:
            cliques, not_all = compute_cliques(g, kwargs.get("max_cliques", 200))
            instance["cliques"] = [set(c) for c in cliques]
            instance["n_cliques"] = len(instance["cliques"])
            instance["all_cliques"] = not not_all
            keys += ["cliques", "n_cliques", "all_cliques"]
        instance["list_edges"] = [[e[0], e[1]] for e in edges]
        keys += ["list_edges"]
        self.instance = instance
        self.dict_datas = {k: instance[k] for k in keys}

    def export_dzn(self, file_name: str = None, keys=None):
        if file_name is None:
            file_name = os.path.join(path_minizinc, "coloring_example_dzn.dzn")
        if keys is None:
            keys = list(self.dict_datas.keys())
        pymzn.dict2dzn(
            {k: self.dict_datas[k] for k in keys if k in self.dict_datas},
            fout=file_name,
        )
        print("Successfully dumped data file ", file_name)

    def retrieve_solutions(
        self, result, parameters_cp: ParametersCP = ParametersCP.default()
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        colors = []
        objectives = []
        solutions_fit = []
        if intermediate_solutions:
            for i in range(len(result)):
                if not self.custom_output_type:
                    colors += [result[i, "color_graph"]]
                    objectives += [result[i, "objective"]]
                else:
                    colors += [result[i].dict["color_graph"]]
                    objectives += [result[i].objective]
        else:
            if not self.custom_output_type:
                colors += [result["color_graph"]]
                objectives += [result["objective"]]
            else:
                colors += [result.dict["color_graph"]]
                objectives += [result.objective]
        for k in range(len(colors)):
            sol = [
                colors[k][self.index_nodes_name[self.nodes_name[i]]] - 1
                for i in range(self.number_of_nodes)
            ]
            color_sol = ColoringSolution(self.coloring_problem, sol)
            fit = self.aggreg_sol(color_sol)
            solutions_fit += [(color_sol, fit)]

        return ResultStorage(
            list_solution_fits=solutions_fit,
            limit_store=False,
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(self, parameters_cp: ParametersCP = None, **kwargs) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        if self.model is None:
            self.init_model(**kwargs)
        limit_time_s = parameters_cp.TimeLimit
        intermediate_solutions = parameters_cp.intermediate_solution
        result = self.instance.solve(
            timeout=timedelta(seconds=limit_time_s),
            intermediate_solutions=intermediate_solutions,
            processes=parameters_cp.nb_process if parameters_cp.multiprocess else None,
            free_search=parameters_cp.free_search,
        )
        verbose = kwargs.get("verbose", False)
        if verbose:
            print("Solving finished")
            print(result.status)
            print(result.statistics)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)

    def get_solution(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.coloring_problem,
                params_objective_function=self.params_objective_function,
            )
            result_store = greedy_solver.solve(
                strategy=kwargs.get("greedy_method", NXGreedyColoringMethod.best),
                verbose=verbose,
            )
            solution = result_store.get_best_solution_fit()[0]
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.coloring_problem.get_dummy_solution()
        return solution

    # Deprecated, better use the generic LNS-CP function !
    def solve_lns(
        self,
        fraction_to_fix: float = 0.9,
        nb_iteration: int = 10,
        parameters_cp: ParametersCP = None,
        **kwargs,
    ):
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        first_solution = self.get_solution(**kwargs)
        dict_color = {
            i + 1: first_solution.colors[i] + 1 for i in range(self.number_of_nodes)
        }
        nb_colors = first_solution.nb_color
        kwargs["nb_colors"] = nb_colors
        self.init_model(**kwargs)
        range_node = range(1, self.number_of_nodes + 1)
        iteration = 0
        current_solution = first_solution
        current_best_solution = current_solution.copy()
        current_nb_color = current_best_solution.nb_color
        while iteration < nb_iteration:
            with self.instance.branch() as child:
                subpart_color = set(
                    random.sample(
                        range_node, int(fraction_to_fix * self.number_of_nodes)
                    )
                )
                for i in range_node:
                    if i in subpart_color and dict_color[i] < current_nb_color - 2:
                        child.add_string(
                            "constraint color_graph["
                            + str(i)
                            + "] == "
                            + str(dict_color[i])
                            + ";\n"
                        )
                    child.add_string(
                        "constraint color_graph["
                        + str(i)
                        + "] <= "
                        + str(current_nb_color)
                        + ";\n"
                    )
                child.add_string(f"solve minimize(obj);\n")
                res = child.solve(
                    timeout=timedelta(seconds=parameters_cp.TimeLimit),
                    intermediate_solutions=parameters_cp.intermediate_solution,
                )
                result_storage = self.retrieve_solutions(
                    res, parameters_cp=parameters_cp
                )
                print(res.status)
                sol, fit = result_storage.get_best_solution_fit()
                nb_color = self.coloring_problem.evaluate(sol)["nb_colors"]
                if res.solution is not None and nb_color < current_nb_color:
                    current_nb_color = nb_color
                    current_best_solution = sol
                    dict_color = {
                        i + 1: current_best_solution.colors[i] + 1
                        for i in range(self.number_of_nodes)
                    }
                    print(iteration, " : , ", fit)
                    print("IMPROVED : ")
                else:
                    try:
                        print(iteration, " : found solution ", nb_color)
                    except:
                        print(iteration, " failed ")
                iteration += 1
        fit = self.coloring_problem.evaluate(current_best_solution)
        return current_best_solution, fit
