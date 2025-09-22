import networkx as nx
from matplotlib import pyplot as plt

from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.allocation.solvers.dp import DpAllocationSolver, dp
from discrete_optimization.workforce.allocation.utils import plot_allocation_solution
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)


def run_dp():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem, multiobjective=False
    )
    solver = DpAllocationSolver(allocation_problem)
    solver.init_model(force_allocation_when_possible=False, symmbreak_on_used=False)
    sol = solver.solve(
        solver=dp.LNBS, time_limit=10, threads=5, retrieve_intermediate_solutions=True
    ).get_best_solution()
    # plot_allocation_solution(problem=allocation_problem, sol=sol)
    # plt.show()


if __name__ == "__main__":
    run_dp()
