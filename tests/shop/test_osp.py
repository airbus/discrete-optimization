#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import pytest

from discrete_optimization.generic_tools.callbacks.loggers import (
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.jsp.parser import get_data_available, parse_file
from discrete_optimization.shop.osp.problem import OpenShopProblem, OpenShopSolution
from discrete_optimization.shop.osp.solvers.cpsat import CpSatOspSolver

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def simple_osp_problem():
    """Create a simple OSP instance with 2 jobs, 3 machines."""
    job_0 = Job(
        job_index=0,
        subjobs=[
            Subjob(0, 0, recipes=[SubjobRecipe(0, 3)]),
            Subjob(1, 0, recipes=[SubjobRecipe(1, 2)]),
            Subjob(2, 0, recipes=[SubjobRecipe(2, 4)]),
        ],
    )
    job_1 = Job(
        job_index=1,
        subjobs=[
            Subjob(0, 1, recipes=[SubjobRecipe(0, 2)]),
            Subjob(1, 1, recipes=[SubjobRecipe(1, 3)]),
            Subjob(2, 1, recipes=[SubjobRecipe(2, 1)]),
        ],
    )
    return OpenShopProblem(list_jobs=[job_0, job_1], n_jobs=2, n_machines=3, horizon=20)


def test_osp_problem_creation(simple_osp_problem):
    """Test that OSP problem is created correctly."""
    problem = simple_osp_problem
    assert problem.n_jobs == 2
    assert problem.n_machines == 3
    assert len(problem.tasks_list) == 6  # 2 jobs * 3 subjobs each


def test_osp_solution_satisfy(simple_osp_problem):
    """Test that a valid OSP solution satisfies constraints."""
    problem = simple_osp_problem
    # Valid schedule: no overlap on machines, no overlap within jobs
    sol = OpenShopSolution(
        problem=problem,
        schedule=[
            [(0, 3), (3, 5), (5, 9)],  # job 0
            [(3, 5), (0, 3), (9, 10)],  # job 1
        ],
        machine_index=[[0, 1, 2], [0, 1, 2]],
    )
    assert problem.satisfy(sol)


def test_osp_solution_nok_machine_overlap(simple_osp_problem, caplog):
    """Test that overlapping tasks on same machine violate constraints."""
    problem = simple_osp_problem
    # Invalid: job 0 subjob 0 and job 1 subjob 0 both use machine 0 and overlap
    sol = OpenShopSolution(
        problem=problem,
        schedule=[
            [(0, 3), (3, 5), (5, 9)],  # job 0
            [(0, 2), (3, 6), (9, 10)],  # job 1 - overlaps on machine 0
        ],
        machine_index=[[0, 1, 2], [0, 1, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)


def test_osp_solution_nok_job_overlap(simple_osp_problem, caplog):
    """Test that overlapping subjobs within same job violate constraints."""
    problem = simple_osp_problem
    # Invalid: job 0's subjobs overlap in time
    sol = OpenShopSolution(
        problem=problem,
        schedule=[
            [(0, 3), (2, 4), (5, 9)],  # job 0 - subjobs 0 and 1 overlap
            [(3, 5), (0, 3), (9, 10)],  # job 1
        ],
        machine_index=[[0, 1, 2], [0, 1, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)


def test_osp_solution_nok_duration(simple_osp_problem, caplog):
    """Test that incorrect duration violates constraints."""
    problem = simple_osp_problem
    # Invalid: job 0 subjob 0 has duration 2 instead of 3
    sol = OpenShopSolution(
        problem=problem,
        schedule=[
            [(0, 2), (3, 5), (5, 9)],  # job 0 - wrong duration for subjob 0
            [(2, 4), (0, 3), (9, 10)],  # job 1
        ],
        machine_index=[[0, 1, 2], [0, 1, 2]],
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)


def test_osp_evaluate(simple_osp_problem):
    """Test evaluation function returns makespan."""
    problem = simple_osp_problem
    sol = OpenShopSolution(
        problem=problem,
        schedule=[
            [(0, 3), (3, 5), (5, 9)],
            [(3, 5), (0, 3), (9, 10)],
        ],
        machine_index=[[0, 1, 2], [0, 1, 2]],
    )
    evaluation = problem.evaluate(sol)
    assert "makespan" in evaluation
    assert evaluation["makespan"] == 10  # max end time


@pytest.mark.parametrize("jsp_file", get_data_available()[:3])
def test_osp_from_jsp_parser(jsp_file):
    """Test creating OSP from JSP data files."""
    # Parse as JSP first
    jsp_problem = parse_file(jsp_file)

    # Convert to OSP (no precedence constraints)
    osp = OpenShopProblem(
        list_jobs=jsp_problem.list_jobs,
        n_jobs=jsp_problem.n_jobs,
        n_machines=jsp_problem.n_machines,
        horizon=jsp_problem.horizon,
    )

    assert isinstance(osp, OpenShopProblem)
    assert osp.n_jobs == jsp_problem.n_jobs
    assert osp.n_machines == jsp_problem.n_machines


def test_osp_cpsat_solver(simple_osp_problem):
    """Test CP-SAT solver on simple OSP instance."""
    problem = simple_osp_problem
    solver = CpSatOspSolver(problem)
    solver.init_model(use_cpm_for_task_bounds=False, use_energy_constraints=False)

    results = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=5,
        callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO)],
    )

    assert len(results) > 0
    best_solution = results[-1][0]
    assert problem.satisfy(best_solution)
    evaluation = problem.evaluate(best_solution)
    assert evaluation["makespan"] > 0
