from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver
from multibatching.utils import generate_multibatching_problem
import logging
logging.basicConfig(level=logging.INFO)
import os
import json
from multibatching.parse_config_data import parse_json_to_problem
from multibatching.solvers.two_steps import (TwoStepMultibatchingSolver, NetxMultibatchingSolver,
                                             PackingViaBinPacking, GreedyPackingForMultibatching)
import time
def parse_problem(fix_problem: bool = False,
                   harbor_constraint: bool = False,
                  filter_transport_types:bool = False,
                  scale_co2: float = 1):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=harbor_constraint,
                                    scale_co2=scale_co2)

    return problem

def script():
    # 1. Generate the problem instance
    from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=5,
                                             num_products=3,
                                             min_dist=1,
                                             max_dist=5, 
                                             seed=51)

    problem = parse_problem(harbor_constraint=False,
                            filter_transport_types=True, scale_co2=1/1000000)
    solver = ClingconMultibatchingSolver(problem)

    use_shortest_path_heuristic = True 
    sp_tolerance = 0.2  

    solver.init_model(restrict_to_shortest_paths=use_shortest_path_heuristic,
                     shortest_path_tolerance=sp_tolerance) 
    time_limit=100
    print(f"Starting solve with {time_limit}s timeout...")
    result_storage = solver.solve(callbacks=[BasicStatsCallback()],
                                  time_limit=time_limit)
    # 4. Retrieve and display results
    solution, fitness = result_storage.get_best_solution_fit()
    if solution == None:
        print("UNSAT") 
    else:
        pass
        fit = solver.aggreg_from_sol(solution)
 
        pack = GreedyPackingForMultibatching(problem)
        pack.init_from_solution(solution)
        res = pack.solve(time_limit_per_link=2)
        sol_ = res.get_best_solution()

        value = sum(problem.evaluate(sol_).values())
        print("total costs:", value, f"({value:.2e})")

if __name__ == "__main__":
    script()