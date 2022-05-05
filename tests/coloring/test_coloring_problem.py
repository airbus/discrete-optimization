from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)


def test_model_satisfy():
    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    dummy_solution = color_problem.get_dummy_solution()
    print("Dummy ", dummy_solution)
    print("Dummy satisfy ", color_problem.satisfy(dummy_solution))
    print(color_problem.evaluate(dummy_solution))
    bad_solution = ColoringSolution(color_problem, [1] * color_problem.number_of_nodes)
    color_problem.evaluate(bad_solution)
    print("Bad solution satisfy ", color_problem.satisfy(bad_solution))


if __name__ == "__main__":
    test_model_satisfy()
