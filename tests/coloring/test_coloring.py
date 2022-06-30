import random

import numpy as np
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
    solve,
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
from discrete_optimization.generic_tools.ea.ga import DeapMutation, Ga
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)

try:
    pass
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


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


def test_model_satisfy():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    dummy_solution = color_problem.get_dummy_solution()
    assert color_problem.satisfy(dummy_solution)
    color_problem.evaluate(dummy_solution)
    bad_solution = ColoringSolution(color_problem, [1] * color_problem.number_of_nodes)
    color_problem.evaluate(bad_solution)
    assert not color_problem.satisfy(bad_solution)


def test_greedy_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(
        strategy=NXGreedyColoringMethod.connected_sequential, verbose=True
    )
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


def test_greedy_best_coloring():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedyColoring(color_problem, params_objective_function=None)
    result_store = solver.solve(strategy=NXGreedyColoringMethod.best, verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


def test_ga_coloring_1(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors_from0",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        objectives=["nb_colors"],
        objective_weights=[-1],
        max_evals=5000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)


def test_ga_coloring_2(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)
    ga_solver = Ga(
        color_problem,
        encoding="colors",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=5000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)


def test_ga_coloring_3(random_seed):
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arity": 10,
    }

    ga_solver = Ga(
        color_problem,
        encoding=encoding,
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["nb_colors", "nb_violations"],
        objective_weights=[-1, -2],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=5000,
    )
    color_sol = ga_solver.solve().get_best_solution()
    color_problem.evaluate(color_sol)
    assert color_problem.satisfy(color_sol)


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
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


def test_coloring_nsga_2():

    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arity": 10,
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
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_color_lp_gurobi():
    file = [f for f in get_data_available() if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = ColoringLP(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    result_store = solver.solve(parameters_milp=ParametersMilp.default(), verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


if __name__ == "__main__":
    test_solvers()
