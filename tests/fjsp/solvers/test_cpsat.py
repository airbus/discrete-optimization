#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Optional

import pytest

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import FJobShopSolution, Job
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver, FJobShopProblem
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
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
    p = ParametersCp.default()
    res = solver.solve(
        parameters_cp=p, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    sol, _ = res.get_best_solution_fit()
    assert fproblem.satisfy(sol)

    # test prob.task_lists and sol.get_start_stime/end_time
    assert len(problem.tasks_list) == problem.n_all_jobs
    for task in problem.tasks_list:
        print(sol.get_start_time(task), sol.get_end_time(task))


def test_cpsat_fjsp():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    res = solver.solve(
        parameters_cp=p,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
    )
    sol, _ = res.get_best_solution_fit()
    assert problem.satisfy(sol)
    # test prob.task_lists and sol.get_start_stime/end_time
    assert len(problem.tasks_list) == problem.n_all_jobs
    for task in problem.tasks_list:
        print(sol.get_start_time(task), sol.get_end_time(task))


def test_objectives():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    solver.init_model()
    p = ParametersCp.default()

    subtasks = {(0, 0), (2, 2)}
    # max end time subtasks
    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == max(sol.get_end_time(task) for task in subtasks)


def test_mode_constraint():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    assert problem.is_multimode

    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    kwargs_solve = dict(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    task = (0, 0)
    print(problem.get_task_modes(task))
    mode = 0
    sol: FJobShopSolution = solver.solve(**kwargs_solve).get_best_solution()
    assert not (sol.get_mode(task) == mode)

    solver.add_constraint_on_task_mode(task, mode)
    sol: FJobShopSolution = solver.solve(**kwargs_solve).get_best_solution()
    assert sol.get_mode(task) == mode

    mode_nok = 5
    with pytest.raises(ValueError):
        solver.add_constraint_on_task_mode(task, mode_nok)


def test_mode_constraint_monomode():
    filename = "la02"
    filepath = [f for f in jsp_parser.get_data_available() if f.endswith(filename)][0]
    problem: JobShopProblem = jsp_parser.parse_file(filepath)
    fproblem = FJobShopProblem(
        list_jobs=[
            Job(job_id=i, sub_jobs=[[sj] for sj in problem.list_jobs[i]])
            for i in range(problem.n_jobs)
        ],
        n_jobs=problem.n_jobs,
        n_machines=problem.n_machines,
    )
    assert not fproblem.is_multimode

    solver = CpSatFjspSolver(problem=fproblem)
    p = ParametersCp.default()
    kwargs_solve = dict(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    task = (0, 1)
    mode = 0
    solver.init_model()
    solver.add_constraint_on_task_mode(task, mode)
    sol: FJobShopSolution = solver.solve(**kwargs_solve).get_best_solution()
    assert sol.get_mode(task) == mode

    mode_nok = 3
    with pytest.raises(ValueError):
        solver.add_constraint_on_task_mode(task, mode_nok)


@pytest.mark.parametrize(
    "task, start_or_end, sign , time",
    [
        ((0, 1), StartOrEnd.START, SignEnum.LESS, 20),
        ((0, 0), StartOrEnd.END, SignEnum.EQUAL, 120),
    ],
)
def test_task_constraints(task, start_or_end, sign, time):
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    sol: FJobShopSolution = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    # before adding the constraint, not already satisfied
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # add constraint: should be now satisfied
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    sol = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # check constraints can be effectively removed
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_chaining_tasks_constraint():
    task1 = (0, 0)
    task2 = (1, 4)
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    sol: FJobShopSolution = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    # before adding the constraint, not already satisfied
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # add constraint: should be now satisfied
    cstrs = solver.add_constraint_chaining_tasks(task1=task1, task2=task2)
    sol = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    assert sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # check constraints can be effectively removed
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        parameters_cp=p,
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    print(sol.schedule)
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)


def test_cpsat_retrieve_stats():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    res = solver.solve(
        parameters_cp=p,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
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


def test_cpsat_retrieve_stats_via_clb():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default()
    stats_clb = StatsCpsatCallback()
    res = solver.solve(
        callbacks=[stats_clb, NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=p,
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
