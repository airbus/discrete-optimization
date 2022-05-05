from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)


def test_greedy_coloring():
    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(
        strategy=NXGreedyColoringMethod.connected_sequential, verbose=True
    )
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def test_greedy_best_coloring():
    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(strategy=NXGreedyColoringMethod.best, verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_greedy_best_coloring()
