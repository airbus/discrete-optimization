#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from typing import Optional

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.cpsat_solver import (
    CPSatRCPSPSolverCumulativeResource,
    CPSatRCPSPSolverResource,
)


class RetrieveSubRes(Callback):
    def __init__(self):
        self.sol_per_step: list[list[Solution]] = []
        self.all_sols = set()

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.sol_per_step.append(
            [s for s, _ in res.list_solution_fits if s not in self.all_sols]
        )
        self.all_sols.update(set(self.sol_per_step[-1]))
        return False


def check_lexico_order_on_result_storage(
    callback_subres: RetrieveSubRes, solver: LexicoSolver
):
    final_obj = {}
    objectives = solver._objectives
    for i in range(len(objectives)):
        # obj = solver._objectives[i]
        # print("sols ", callback_subres.sol_per_step[i])
        # print("len ", len(callback_subres.sol_per_step))
        obj_array = np.array(
            [
                [sol._internal_objectives[obj] for obj in objectives]
                for sol in callback_subres.sol_per_step[i]
            ]
        )
        final_obj[i] = obj_array[-1, i]
        assert (obj_array[1:, i] <= obj_array[:-1, i]).all()  # decreasing objs
        if i > 0:
            for ii in range(i):
                assert (
                    obj_array[:, ii] <= final_obj[ii]
                ).all()  # Don't degrade previous objectives.


@pytest.mark.parametrize(
    "objectives",
    [
        None,
        ["makespan", "used_resource"],
        ["used_resource", "makespan"],
    ],
)
def test_ortools_cumulativeresource_optim(objectives):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    subsolver = CPSatRCPSPSolverCumulativeResource(problem=rcpsp_problem)

    solver = LexicoSolver(
        problem=rcpsp_problem,
        subsolver=subsolver,
    )
    solver.init_model()
    clb = RetrieveSubRes()
    result_storage = solver.solve(time_limit=10, objectives=objectives, callbacks=[clb])

    print([sol._internal_objectives for sol, fit in result_storage.list_solution_fits])
    check_lexico_order_on_result_storage(callback_subres=clb, solver=solver)


@pytest.mark.parametrize(
    "objectives",
    [
        None,
        ["makespan", "used_resource"],
        ["used_resource", "makespan"],
    ],
)
def test_ortools_resource_optim(objectives):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    subsolver = CPSatRCPSPSolverResource(problem=rcpsp_problem)

    solver = LexicoSolver(
        problem=rcpsp_problem,
        subsolver=subsolver,
    )
    solver.init_model()
    clb = RetrieveSubRes()
    result_storage = solver.solve(
        callbacks=[clb],
        time_limit=10,
        objectives=objectives,
    )
    print([sol._internal_objectives for sol, fit in result_storage.list_solution_fits])
    check_lexico_order_on_result_storage(callback_subres=clb, solver=solver)
