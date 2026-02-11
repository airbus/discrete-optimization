#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

#  Author(s): Hieu Tran

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee, SkillDetail, VariantMultiskillRcpspProblem
)
from discrete_optimization.rcpsp_multiskill.solvers.multimode_transposition import (
    MultimodeTranspositionMultiskillRcpspSolver,
)

logging.basicConfig(level=logging.INFO)


def create_toy_msrcpsp_medium():
    """Create a medium-complexity toy multi-skill RCPSP problem for testing."""
    import numpy as np
    
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2"}
    non_renewable_resources = set()
    resources_availability = {
        "R1": np.array([5] * 100, dtype=int),
        "R2": np.array([4] * 100, dtype=int),
    }
    
    # Create 3 employees with different skill combinations
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
    
    # Task structure with single mode per task (to avoid complexity)
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

def run_multimode_transposition_on_toy_problem():
    """
    Run the multimode transposition solver on a toy multi-skill RCPSP instance.
    """
    print("\n=== Example on toy instance ===")

    # Create a toy problem
    model = create_toy_msrcpsp_medium()
    print(f"Problem info: "
          f"{len(model.tasks_list)} tasks, "
          f"{len(model.employees)} employees, "
          f"{len(model.skills_set)} skills, "
          f"and {len(model.resources_set)} resources.")

    # Create the solver
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=model,
        reconstruction_cp_time_limit=60,
    )

    # Configure solver parameters
    parameters_cp = ParametersCp.default()
    parameters_cp.intermediate_solution = True  # Store intermediate solutions

    # Solve the problem
    print(f"Solving the toy problem with multimode transposition approach...")
    result_store = solver.solve(parameters_cp=parameters_cp, time_limit=30)
    
    # Retrieve the best solution
    best_solution, fit = result_store.get_best_solution_fit()
    if best_solution is None:
        print("No solution found within the time limit.")
    else:
        print("SOLUTION FOUND")
        print(f"Objective/Fitness: {fit}")
        print(f"Makespan: {best_solution.get_end_time(model.sink_task)}")
        print(f"Feasible: {model.satisfy(best_solution)}")

def run_multimode_transposition_on_imopse():
    """
    Run the multimode transposition solver on an IMOPSE instance.
    """
    print("\n=== Example on impose instance ===")
    # Get available IMOPSE files
    files = get_data_available()
    
    if not files:
        print("No IMOPSE files found in the data directory.")
        print("Please ensure the discrete_optimization_data is available.")
        return
    
    # Find the small instance or use the first available file
    test_file = None
    for f in files:
        if "100_5_64_9" in f:
            test_file = f
            break
    
    if test_file is None:
        # Use the first available file if not found
        test_file = files[0]
        print(f"Instance used: {test_file.split('/')[-1]}")
    
    # Parse the file
    imopse_problem, _ = parse_file(test_file, max_horizon=1000)

    print(f"\nLoading IMOPSE instance: {test_file.split('/')[-1]}")
    print(f"  - Tasks: {len(imopse_problem.tasks_list)}")
    print(f"  - Employees: {len(imopse_problem.employees)}")
    print(f"  - Skills: {len(imopse_problem.skills_set)}")
    print(f"  - Resources: {list(imopse_problem.resources_set)}")
    print(f"  - Horizon: {imopse_problem.horizon}")
    
    # Calculate Lower Bound = minimum theoretical makespan (sum of critical path)
    # ie. ignoring resource constraints, all tasks in parallel
    total_min_duration = sum(
        imopse_problem.mode_details[task][1].get('duration', 0)
        for task in imopse_problem.tasks_list
    )
    print(f"  - Lower bound: {total_min_duration}")
    
    RECONSTRUCTION_TIME_LIMIT = 120  # time limit (s) for the CP-based reconstruction
    MULTIMODE_SOLVER_TIME_LIMIT = 60  # time limit (s) for the multimode solver

    # Create the solver for the IMOPSE instance
    solver = MultimodeTranspositionMultiskillRcpspSolver(
        problem=imopse_problem,
        reconstruction_cp_time_limit=RECONSTRUCTION_TIME_LIMIT,
    )
    
    # Configure solver parameters
    parameters_cp = ParametersCp.default()
    parameters_cp.intermediate_solution = True  # Store intermediate solutions
    
    # Solve the problem
    print(f"\nSolving with multimode transposition approach...")
    print(f"  - Multimode solver time limit: {MULTIMODE_SOLVER_TIME_LIMIT} seconds")
    print(f"  - Reconstruction CP time limit: {RECONSTRUCTION_TIME_LIMIT} seconds")
    result_store = solver.solve(parameters_cp=parameters_cp, time_limit=MULTIMODE_SOLVER_TIME_LIMIT)
    
    # Retrieve the best solution
    best_solution, fit = result_store.get_best_solution_fit()
    
    if best_solution is None:
        print("\nNo solution found within the time limit.")
    else:
        print("SOLUTION FOUND")
        print(f"Objective/Fitness: {fit}")
        print(f"Makespan: {best_solution.get_end_time(imopse_problem.sink_task)}")
        print(f"Feasible: {imopse_problem.satisfy(best_solution)}")
        

if __name__ == "__main__":
    run_multimode_transposition_on_toy_problem()
    run_multimode_transposition_on_imopse()
