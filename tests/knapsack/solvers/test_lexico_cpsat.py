#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, MobjKnapsackModel
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver


def fit2dict(solver: SolverDO, fit: TupleFitness) -> dict[str, float]:
    return dict(zip(solver.problem.get_objective_names(), fit.vector_fitness))


class MyCallback(Callback):
    def __init__(self):
        self.res_by_step: list[list[dict[str, float]]] = []

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        subres = [fit2dict(solver, fit) for sol, fit in res.list_solution_fits]
        print(subres)
        self.res_by_step.append(subres)

        return False


@pytest.mark.parametrize(
    "objectives",
    [
        ("value", "weight", "heaviest_item"),
        ("weight", "heaviest_item", "value"),
        ("heaviest_item", "value", "weight"),
    ],
)
def test_knapsack_ortools_lexico(objectives):
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    model: MobjKnapsackModel = MobjKnapsackModel.from_knapsack(model)
    subsolver = CpSatKnapsackSolver(model)
    solver = LexicoSolver(
        problem=model,
        subsolver=subsolver,
    )
    solver.init_model()
    mycb = MyCallback()
    parameters_cp = ParametersCp.default()
    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=parameters_cp,
        objectives=objectives,
        callbacks=[mycb],
    )

    previous_obj_best_value = None
    previous_obj = None
    for obj, subres in zip(objectives, mycb.res_by_step):
        if previous_obj_best_value is not None and previous_obj is not None:
            assert all(d[previous_obj] >= previous_obj_best_value for d in subres)
        previous_obj_best_value = max([d[obj] for d in subres])
        previous_obj = obj

    last_fit_dict = mycb.res_by_step[-1][-1]
    first_obj = objectives[0]
    if first_obj == "value":
        assert last_fit_dict[first_obj] == 99837.0
    elif first_obj == "weight":
        assert last_fit_dict[first_obj] == 0.0
    elif first_obj == "heaviest_item":
        assert last_fit_dict[first_obj] == 90001.0
