from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    TypeAttribute,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)


def test_ga_coloring_1():
    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors_from0",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=30000,
    )
    color_sol = ga_solver.solve()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


def test_ga_coloring_2():
    file = [f for f in files_available if "gc_100_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=100000,
    )
    color_sol = ga_solver.solve()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


def test_ga_coloring_3():
    file = [f for f in files_available if "gc_100_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "custom": {
            "name": "colors",
            "type": [TypeAttribute.LIST_INTEGER],
            "n": 70,
            "arrity": 15,
        }
    }
    ga_solver = Ga(
        color_problem,
        encoding=encoding,
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=100000,
    )
    color_sol = ga_solver.solve()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


if __name__ == "__main__":
    test_ga_coloring_3()
