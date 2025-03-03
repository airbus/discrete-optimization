#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Optional

import pytest

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import Job
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver, FJobShopProblem
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.jsp.problem import JobShopProblem

logging.basicConfig(level=logging.INFO)


class StatsCpsatCallback(Callback):
    def __init__(self):
        self.starting_time: int = None
        self.end_time: int = None
        self.stats: list[dict] = []
        self.final_status: str = None

    def on_step_end(
        self, step: int, res: ResultStorage, solver: OrtoolsCpSatSolver
    ) -> Optional[bool]:
        self.stats.append(
            {
                "obj": solver.clb.ObjectiveValue(),
                "bound": solver.clb.BestObjectiveBound(),
                "time": time.perf_counter() - self.starting_time,
                "time-cpsat": {
                    "user-time": solver.clb.UserTime(),
                    "wall-time": solver.clb.WallTime(),
                },
            }
        )
        if solver.clb.ObjectiveValue() == solver.clb.BestObjectiveBound():
            return False

    def on_solve_start(self, solver: OrtoolsCpSatSolver):
        self.starting_time = time.perf_counter()

    def on_solve_end(self, res: ResultStorage, solver: OrtoolsCpSatSolver):
        """Called at the end of solve.
        Args:
        res: current result storage
        solver: solvers using the callback
        """
        status_name = solver.solver.status_name()
        # status_do: StatusSolver = cpstatus_to_dostatus(status_name)
        if len(self.stats) > 0:
            if (
                solver.solver.ObjectiveValue() != self.stats[-1]["obj"]
                or solver.solver.BestObjectiveBound() != self.stats[-1]["bound"]
            ):
                self.stats.append(
                    {
                        "obj": solver.solver.ObjectiveValue(),
                        "bound": solver.solver.BestObjectiveBound(),
                        "time": time.perf_counter() - self.starting_time,
                        "time-cpsat": {
                            "user-time": solver.solver.UserTime(),
                            "wall-time": solver.solver.WallTime(),
                        },
                    }
                )
        self.final_status = status_name


def test_fjsp_solver_on_jsp():
    file_path = jsp_parser.get_data_available()[1]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem: JobShopProblem = jsp_parser.parse_file(file_path)
    fproblem = FJobShopProblem(
        list_jobs=[
            Job(job_id=i, sub_jobs=[[sj] for sj in problem.list_jobs[i]])
            for i in range(problem.n_jobs)
        ],
        n_jobs=problem.n_jobs,
        n_machines=problem.n_machines,
    )
    solver = CpSatFjspSolver(problem=fproblem)
    p = ParametersCp.default_cpsat()
    res = solver.solve(parameters_cp=p, time_limit=20)
    sol, _ = res.get_best_solution_fit()
    assert fproblem.satisfy(sol)


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_cpsat_fjsp():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=30,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
    )
    sol, _ = res.get_best_solution_fit()
    assert problem.satisfy(sol)


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_cpsat_retrieve_stats():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=5,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        retrieve_stats=True,
    )
    assert res.stats is not None
    assert res.stats[-1]["obj"] == solver.solver.ObjectiveValue()
    # assert res.stats[-1]["bound"] == solver.solver.BestObjectiveBound()
    sol, _ = res.get_best_solution_fit()
    assert problem.satisfy(sol)


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_cpsat_retrieve_stats_via_clb():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    stats_clb = StatsCpsatCallback()
    res = solver.solve(
        callbacks=[stats_clb],
        parameters_cp=p,
        time_limit=20,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        retrieve_stats=False,
    )
    assert stats_clb.stats is not None
    assert stats_clb.stats[-1]["obj"] == solver.solver.ObjectiveValue()
    print(stats_clb.stats)
    # assert res.stats[-1]["bound"] == solver.solver.BestObjectiveBound()
    sol, _ = res.get_best_solution_fit()
    assert problem.satisfy(sol)
