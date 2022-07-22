from copy import deepcopy

from discrete_optimization.pickup_vrp.solver.ortools_solver import (
    GPDP,
    ORToolsGPDP,
    name_firstsolution_to_value,
    name_metaheuristic_to_value,
)


def run(X):
    s: ORToolsGPDP = X[0]
    res = s.solve(X[1])
    if len(res) > 0:
        print(min(res, key=lambda x: x[-1])[-1], X[2])
    return res, X[2]


class BenchmarkOrtoolsUtility:
    def __init__(self, problem: GPDP):
        self.problem = problem

    def init_solvers(self, **kwargs):
        solvers_list = []
        for method_metaheuristic in name_metaheuristic_to_value:
            for method_first_solution in name_firstsolution_to_value:
                for use_lns in [True, False]:
                    for use_cp in [True, False]:
                        solver = ORToolsGPDP(problem=self.problem)
                        dict_params = deepcopy(kwargs)
                        dict_params[
                            "first_solution_strategy"
                        ] = name_firstsolution_to_value[method_first_solution]
                        dict_params[
                            "local_search_metaheuristic"
                        ] = name_metaheuristic_to_value[method_metaheuristic]
                        dict_params["use_lns"] = use_lns
                        dict_params["use_cp"] = use_cp
                        solver.init_model(**dict_params)
                        params = solver.build_search_parameters(**dict_params)
                        solvers_list += [
                            (
                                solver,
                                params,
                                {
                                    "local_search_metaheuristic": method_metaheuristic,
                                    "first_solution_strategy": method_first_solution,
                                    "use_lns": use_lns,
                                    "use_cp": use_cp,
                                },
                            )
                        ]
        self.solvers_list = solvers_list
        return solvers_list

    def run_benchmark(self):
        results = map(run, self.solvers_list)
        ps = []
        for res, p in results:
            if len(res) > 0:
                best_results = min(res, key=lambda x: x[-1])
                ps += [(best_results, best_results[-1], p)]
            else:
                ps += [(None, None, p)]
        return ps
