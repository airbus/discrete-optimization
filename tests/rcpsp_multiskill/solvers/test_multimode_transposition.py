#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee,
    MultiskillRcpspSolution,
    SkillDetail,
    VariantMultiskillRcpspProblem,
)
from discrete_optimization.rcpsp_multiskill.solvers.multimode_transposition import (
    MultimodeTranspositionMultiskillRcpspSolver
)


def assert_solver_initialized(solver, model):
    """Assert that the solver is properly initialized before solving."""
    assert solver.problem == model
    assert solver.solver_multimode_rcpsp is None
    assert solver.multimode_problem is None
    assert solver.worker_type_to_worker is None


def assert_solver_solved(solver):
    """Assert that the solver has been properly executed."""
    assert solver.multimode_problem is not None
    assert solver.worker_type_to_worker is not None
    assert isinstance(solver.worker_type_to_worker, dict)
    for worker_type, workers in solver.worker_type_to_worker.items():
        assert isinstance(worker_type, str)
        assert isinstance(workers, set)


def assert_valid_result_storage(result_storage):
    """Assert that result storage contains valid solutions."""
    assert result_storage is not None, "Solver did not return a result"
    assert result_storage.list_solution_fits is not None, "Solver did not return solution fits"
    assert len(result_storage.list_solution_fits) > 0, "Solver did not return any solution fits"


def assert_valid_solution(model, solution):
    """Assert that the solution is valid and satisfies all constraints."""
    assert solution is not None, "Solver did not return a solution"
    assert isinstance(solution, MultiskillRcpspSolution), "Best solution is not of type MultiskillRcpspSolution"
    
    # Evaluate and verify constraints
    objective = model.evaluate(solution)
    assert objective is not None
    assert model.satisfy(solution), "Solution does not satisfy problem constraints"
    
    # Verify schedule
    assert solution.schedule is not None
    assert len(solution.schedule) > 0
    
    # Check all non-dummy tasks are scheduled
    for task in model.tasks_list:
        if task not in [model.source_task, model.sink_task]:
            assert task in solution.schedule, f"Task {task} not scheduled"
    
    # Verify makespan
    sink_end_time = solution.get_end_time(model.sink_task)
    assert sink_end_time > 0, "Sink task end time should be positive"
    assert sink_end_time <= model.horizon, f"Sink end time {sink_end_time} exceeds horizon {model.horizon}"
    
    # Verify employee usage
    assert hasattr(solution, 'employee_usage')
    assert solution.employee_usage is not None
    
    # Check employee assignments for tasks requiring skills
    for task in solution.schedule:
        task_details = model.mode_details[task][1]
        task_needs_employees = any(skill in model.skills_set for skill in task_details.keys())
        
        if task_needs_employees and task not in [model.source_task, model.sink_task]:
            if task in solution.employee_usage:
                assert len(solution.employee_usage[task]) > 0


def solve_and_validate(model, solver, time_limit=60):
    """Solve the problem and validate the solution."""
    assert_solver_initialized(solver, model)
    
    parameters_cp = ParametersCp.default()
    parameters_cp.intermediate_solution = True
    result_storage = solver.solve(parameters_cp=parameters_cp, time_limit=time_limit)
    
    assert_solver_solved(solver)
    assert_valid_result_storage(result_storage)
    
    best_solution, best_fit = result_storage.get_best_solution_fit()
    assert_valid_solution(model, best_solution)
    
    return result_storage, best_solution


def create_toy_msrcpsp_simple():    
    skills_set: set[str] = {"S1", "S2"}
    resources_set: set[str] = {"R1"}
    non_renewable_resources = set()
    resources_availability = {"R1": np.array([10] * 100, dtype=int)}
    
    # 2 employees with different skills
    employee: dict[int, Employee] = {
        1: Employee(
            dict_skill={
                "S1": SkillDetail(10, 0, 0),
                "S2": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(10, 0, 0)},
            calendar_employee=[True] * 100,
        ),
    }
    
    employees_availability: list[int] = [2] * 100
    
    mode_details: dict[int, dict[int, dict[str, int]]] = {
        1: {1: {"R1": 0, "duration": 0}},
        2: {1: {"S1": 1, "R1": 1, "duration": 2}},
        3: {1: {"S2": 1, "R1": 1, "duration": 3}},
        4: {1: {"R1": 0, "duration": 0}},
    }
    
    successors: dict[int, list[int]] = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [],
    }
    
    model = VariantMultiskillRcpspProblem(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    return model


def create_toy_msrcpsp_medium():
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2"}
    non_renewable_resources = set()
    resources_availability = {
        "R1": np.array([5] * 100, dtype=int),
        "R2": np.array([4] * 100, dtype=int),
    }
    
    # 3 employees with different skill combinations
    employee: dict[int, Employee] = {
        1: Employee(
            dict_skill={
                "S1": SkillDetail(10, 0, 0),
                "S2": SkillDetail(10, 0, 0),
                "S3": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={
                "S2": SkillDetail(10, 0, 0),
                "S3": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(10, 0, 0)},
            calendar_employee=[True] * 100,
        ),
    }
    
    employees_availability: list[int] = [3] * 100
    
    mode_details: dict[int, dict[int, dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "duration": 0}},
        2: {1: {"S1": 1, "R1": 2, "R2": 0, "duration": 2}},
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "duration": 5}},
        5: {1: {"S1": 1, "S3": 1, "R1": 1, "R2": 1, "duration": 3}},
        6: {1: {"R1": 0, "R2": 0, "duration": 0}},
    }
    
    successors: dict[int, list[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6],
        6: [],
    }
    
    model = VariantMultiskillRcpspProblem(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    return model


def test_solve_simple():
    """Test solving a simple multi-skill RCPSP problem."""
    model = create_toy_msrcpsp_simple()
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=model,
        solver_multimode_rcpsp=None,
    )
    solve_and_validate(model, solver)


def test_solve_medium():
    """Test solving a medium-complexity multi-skill RCPSP problem."""
    model = create_toy_msrcpsp_medium()
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=model,
        solver_multimode_rcpsp=None,
    )
    solve_and_validate(model, solver)

@pytest.mark.slow
def test_with_imopse_data():
    """Test the solver with real IMOPSE benchmark data."""
    try:
        from discrete_optimization.rcpsp_multiskill.parser_imopse import (
            get_data_available,
            parse_file,
        )
    except ImportError:
        pytest.skip("IMOPSE parser not available")
    
    # Get available IMOPSE files
    files = get_data_available()
    if not files:
        pytest.skip("No IMOPSE data files available")
    
    # Use a small file for testing
    test_file = next((f for f in files if "100_5_20_9" in f), files[0])
    
    # Parse the file
    model, _ = parse_file(test_file, max_horizon=500)
    
    # Create and solve
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=model,
        solver_multimode_rcpsp=None,
    )
    solve_and_validate(model, solver)


def test_solver_custom_parameters():
    """Test that the solver accepts and uses custom parameters."""
    model = create_toy_msrcpsp_simple()
    
    # Create solver with custom parameters
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=model,
        solver_multimode_rcpsp=None,
        limit_number_of_mode_per_task=False,
        max_number_of_mode=5,
        check_resource_compliance=False,
        reconstruction_cp_time_limit=120,
    )
    
    # Verify parameters are stored
    assert solver.limit_number_of_mode_per_task is False
    assert solver.max_number_of_mode == 5
    assert solver.check_resource_compliance is False
    assert solver.reconstruction_cp_time_limit == 120
    
    # Solve and verify it works
    solve_and_validate(model, solver)


if __name__ == "__main__":
    test_solve_simple()
    test_solve_medium()
    test_solver_custom_parameters()
    test_with_imopse_data()
