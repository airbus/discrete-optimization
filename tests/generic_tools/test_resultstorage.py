#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import pytest

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import TrivialSolverFromResultStorage
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


@pytest.fixture
def problem():
    nodes = [(i, {}) for i in range(4)]
    edges = [(0, 1, {}), (0, 2, {}), (1, 3, {}), (2, 3, {})]
    return ColoringProblem(graph=Graph(nodes=nodes, edges=edges))


def test_get_best_solution_fit(problem):
    list_solution_fits = [
        (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), 1),
        (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), 5),
        (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), 2),
    ]
    res = ResultStorage(
        mode_optim=ModeOptim.MINIMIZATION, list_solution_fits=list_solution_fits
    )
    assert res.get_best_solution_fit(satisfying=problem)[1] == 2
    assert res.get_best_solution_fit()[1] == 1

    list_solution_fits = [
        (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), -1),
        (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), -5),
        (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), -2),
    ]
    res = ResultStorage(
        mode_optim=ModeOptim.MAXIMIZATION, list_solution_fits=list_solution_fits
    )
    assert res.get_best_solution_fit(satisfying=problem)[1] == -2
    assert res.get_best_solution_fit()[1] == -1


def test_solver_from_result_storage(problem):
    list_solution_fits = [
        (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), 1),
        (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), 5),
        (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), 2),
    ]
    res = ResultStorage(
        mode_optim=ModeOptim.MINIMIZATION, list_solution_fits=list_solution_fits
    )
    solver = TrivialSolverFromResultStorage(problem=problem, result_storage=res)
    res2 = solver.solve()

    assert res == res2


def test_mutablesequence_behaviour(problem):

    # default value for list_solution_fits
    res = ResultStorage(mode_optim=ModeOptim.MAXIMIZATION)
    # mesuring length of res
    assert len(res) == 0
    # appending to res
    res.append((ColoringSolution(colors=[0, 0, 0, 0], problem=problem), -1))
    assert len(res) == 1
    # extending res
    res.extend(
        [
            (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), -5),
            (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), -2),
        ]
    )
    assert len(res) == 3

    # iterating other res
    assert list(res) == res.list_solution_fits

    # poping res
    sol, fit = res.pop()
    assert fit == -2

    # getting item
    sol, fit = res[0]
    assert fit == -1

    # setting item
    res[0] = sol, -4
    assert res.list_solution_fits[0][1] == -4


def test_add_resultstorage_nok_mode_optim():
    res1 = ResultStorage(
        mode_optim=ModeOptim.MAXIMIZATION,
        list_solution_fits=[
            (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), -1)
        ],
    )
    res2 = ResultStorage(
        mode_optim=ModeOptim.MINIMIZATION,
        list_solution_fits=[
            (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), 5),
            (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), 2),
        ],
    )
    with pytest.raises(ValueError):
        res = res1 + res2


def test_add_resultstorage():
    res1 = ResultStorage(
        mode_optim=ModeOptim.MAXIMIZATION,
        list_solution_fits=[
            (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), -1)
        ],
    )
    res2 = ResultStorage(
        mode_optim=ModeOptim.MAXIMIZATION,
        list_solution_fits=[
            (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), -5),
            (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), -2),
        ],
    )

    res = res1 + res2
    assert isinstance(res, ResultStorage)
    assert len(res) == 3
    assert res[-1][1] == -2
