#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


def test_get_best_solution_fit():
    nodes = [(i, {}) for i in range(4)]
    edges = [(0, 1, {}), (0, 2, {}), (1, 3, {}), (2, 3, {})]
    problem = ColoringProblem(graph=Graph(nodes=nodes, edges=edges))

    list_solution_fits = [
        (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), 1),
        (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), 5),
        (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), 2),
    ]
    res = ResultStorage(
        list_solution_fits=list_solution_fits,
        mode_optim=ModeOptim.MINIMIZATION,
    )
    assert res.get_best_solution_fit(satisfying=problem)[1] == 2
    assert res.get_best_solution_fit()[1] == 1

    list_solution_fits = [
        (ColoringSolution(colors=[0, 0, 0, 0], problem=problem), -1),
        (ColoringSolution(colors=[0, 1, 2, 3, 4], problem=problem), -5),
        (ColoringSolution(colors=[0, 1, 1, 0, 0], problem=problem), -2),
    ]
    res = ResultStorage(
        list_solution_fits=list_solution_fits,
        mode_optim=ModeOptim.MAXIMIZATION,
    )
    assert res.get_best_solution_fit(satisfying=problem)[1] == -2
    assert res.get_best_solution_fit()[1] == -1
