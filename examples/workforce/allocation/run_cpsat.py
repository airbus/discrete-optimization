import networkx as nx

from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)


def run_cpsat():
    instances = [p for p in get_data_available()]
    scheduling_problem = parse_json_to_problem(instances[1])
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(
        time_limit=5, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    ).get_best_solution()


if __name__ == "__main__":
    run_cpsat()
