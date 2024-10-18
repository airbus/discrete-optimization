#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import (
    ColoringProblem,
    transform_color_values_to_value_precede_on_other_node_order,
)
from discrete_optimization.coloring.solvers.dp import (
    DpColoringModeling,
    DpColoringSolver,
    dp,
)
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)


@pytest.mark.parametrize(
    "modeling",
    [DpColoringModeling.COLOR_TRANSITION, DpColoringModeling.COLOR_NODE_TRANSITION],
)
def test_coloring_dp(modeling):
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem: ColoringProblem = parse_file(small_example)
    solver = DpColoringSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, modeling=modeling, time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


@pytest.mark.parametrize(
    "modeling",
    [DpColoringModeling.COLOR_TRANSITION, DpColoringModeling.COLOR_NODE_TRANSITION],
)
def test_dp_coloring_ws(modeling):
    file = [f for f in get_data_available() if "gc_20_7" in f][0]
    color_problem = parse_file(file)
    greedy = GreedyColoringSolver(color_problem)
    sol, _ = greedy.solve(strategy=NxGreedyColoringMethod.best).get_best_solution_fit()
    solver = DpColoringSolver(color_problem)
    solver.init_model(modeling=modeling, nb_colors=30)
    solver.set_warm_start(sol)
    result_store = solver.solve(
        solver=dp.DDLNS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        threads=1,
        retrieve_intermediate_solutions=True,
        time_limit=100,
    )
    solution, fit = result_store.get_best_solution_fit()
    trans_ws = transform_color_values_to_value_precede_on_other_node_order(
        sol.colors, nodes_ordering=solver.nodes_reordering
    )
    print(trans_ws, solution.colors)
    assert solution.colors == trans_ws
    assert color_problem.satisfy(solution)


class MyCallbackNok(Callback):
    def on_step_end(self, step: int, res, solver):
        raise RuntimeError("Explicit crash")


def test_coloring_dp_callback_nok():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem: ColoringProblem = parse_file(small_example)
    solver = DpColoringSolver(problem=problem)
    with pytest.raises(RuntimeError, match="Explicit crash"):
        solver.solve(solver=dp.LNBS, time_limit=5, callbacks=[MyCallbackNok()])


def test_coloring_dp_callback_stop():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem: ColoringProblem = parse_file(small_example)
    solver = DpColoringSolver(problem=problem)

    kwargs = dict(solver=dp.CABS, time_limit=5)
    result_store = solver.solve(**kwargs)
    assert len(result_store) > 1

    stopper = NbIterationStopper(nb_iteration_max=1)
    result_store = solver.solve(callbacks=[stopper], **kwargs)
    assert len(result_store) == 1
