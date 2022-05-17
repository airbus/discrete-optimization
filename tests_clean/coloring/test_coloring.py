import random
from typing import Dict, Hashable, Tuple

import pytest
from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.coloring_solvers import (
    ColoringLP,
    ColoringLP_MIP,
    look_for_solver,
    look_for_solver_class,
    solve,
    solvers_compatibility,
    solvers_map,
)
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
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
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.parametrize("coloring_problem_file", get_data_available())
def test_load_file(coloring_problem_file):
    coloring_model: ColoringProblem = parse_file(coloring_problem_file)
    dummy_solution = coloring_model.get_dummy_solution()
    assert coloring_model.satisfy(dummy_solution)


def test_solvers():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    coloring_model: ColoringProblem = parse_file(small_example)
    assert coloring_model.graph is not None
    assert coloring_model.number_of_nodes is not None
    assert coloring_model.graph.nodes_name is not None
    solvers = solvers_map.keys()
    for s in solvers:
        if s == ColoringLP and not gurobi_available:
            # you need a gurobi licence to test this solver.
            continue
        results = solve(method=s, coloring_model=coloring_model, **solvers_map[s][1])
        s, f = results.get_best_solution_fit()
        print(s, f)


def test_model_satisfy():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    dummy_solution = color_problem.get_dummy_solution()
    print("Dummy ", dummy_solution)
    print("Dummy satisfy ", color_problem.satisfy(dummy_solution))
    print(color_problem.evaluate(dummy_solution))
    bad_solution = ColoringSolution(color_problem, [1] * color_problem.number_of_nodes)
    color_problem.evaluate(bad_solution)
    print("Bad solution satisfy ", color_problem.satisfy(bad_solution))


def test_greedy_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(
        strategy=NXGreedyColoringMethod.connected_sequential, verbose=True
    )
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def test_greedy_best_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(strategy=NXGreedyColoringMethod.best, verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def test_ga_coloring_1():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors_from0",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        objectives=["nb_colors"],
        objective_weights=[-1],
        max_evals=3000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


def test_ga_coloring_2():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


def test_ga_coloring_3():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arrity": 10,
    }

    ga_solver = Ga(
        color_problem,
        encoding=encoding,
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    print("color_sol: ", color_sol)
    print("color_evaluate: ", color_problem.evaluate(color_sol))
    print("color_satisfy: ", color_problem.satisfy(color_sol))


def test_coloring_nsga_1():

    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    objectives = ["nb_colors", "nb_violations"]
    ga_solver = Nsga(
        color_problem,
        encoding="colors",
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


def test_coloring_nsga_2():

    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arrity": 10,
    }

    objectives = ["nb_colors", "nb_violations"]
    ga_solver = Nsga(
        color_problem,
        encoding=encoding,
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


def test_color_lp_gurobi():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = ColoringLP(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    result_store = solver.solve(parameters_milp=ParametersMilp.default(), verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_solvers()
