#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example of using the CP-SAT solver for the Oven Scheduling Problem."""

import logging

from example_utils import (
    load_problem,
    visualize_solution,
)

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.ovensched.problem import OvenSchedulingSolution
from discrete_optimization.ovensched.solution_vector import VectorOvenSchedulingSolution
from discrete_optimization.ovensched.solvers.cpsat_auto import (
    OvenSchedulingAutoCpSatSolver,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def solve_with_cpsat(
    problem, time_limit: int = 100, solution: OvenSchedulingSolution = None
):
    """Solve with CP-SAT solver."""
    solver = OvenSchedulingAutoCpSatSolver(problem=problem)
    max_nb_batch = None
    if solution:
        if isinstance(solution, VectorOvenSchedulingSolution):
            solution = solution.to_oven_scheduling_solution()
        max_batches_in_solution = solution.get_max_nb_batch_per_machine()
        max_nb_batch = max(1, int(max_batches_in_solution * 1.2))
        print(
            f"Warm-start has max {max_batches_in_solution} batches/machine, using {max_nb_batch} for model"
        )
    solver.init_model()
    params = ParametersCp.default_cpsat()
    params.nb_process = 12

    if solution:
        solver.set_warm_start(solution)

    result = solver.solve(
        callbacks=[ProblemEvaluateLogger(step_verbosity_level=logging.INFO)],
        time_limit=time_limit,
        parameters_cp=params,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    for key in solver.batch_attr_vars:
        print(key, ":", solver.solver.value(solver.batch_attr_vars[key]))
    for i in range(len(problem.ordered_routing_entities)):
        for ent in problem.ordered_routing_entities[i].list_entities:
            for attr in solver.attribute_variable_routing[i][ent]:
                val = solver.solver.value(
                    solver.attribute_variable_routing[i][ent][attr]
                )
                if hasattr(ent, "batch_id"):
                    if ent.batch_id in solver.batch_attr_vars:
                        if val:
                            print(
                                i,
                                ent.entity_id,
                                "attr",
                                attr,
                                " vs ",
                                solver.solver.value(
                                    solver.batch_attr_vars[ent.batch_id]
                                ),
                            )
    for task in solver.task_to_batch_vars:
        for batch in solver.task_to_batch_vars[task]:
            val = solver.solver.value(solver.task_to_batch_vars[task][batch])
            if val:
                print(
                    "task",
                    task,
                    "batch",
                    batch,
                    "attribute",
                    problem.get_task_attribute(task),
                    " is in batch ",
                )
    for index in solver.cumul_addition_variable_routing:
        for c in solver.cumul_addition_variable_routing[index]:
            for ent in solver.cumul_addition_variable_routing[index][c]:
                print(
                    "addition cost ",
                    ent.entity_id,
                    solver.solver.value(
                        solver.cumul_addition_variable_routing[index][c][ent]
                    ),
                )
    return result, solver


def run_cpsat_example():
    """Run a simple example with the CP-SAT solver."""
    instance_name = "50NewRandomOvenSchedulingInstance-n50-k2-a5--2904-15.05.46.dat"
    # instance_name = "11RandomOvenSchedulingInstance-n10-k5-a2-WithInitialStates.dat"
    # instance_name = "32RandomOvenSchedulingInstance-n25-k5-a2-WithInitialStates.dat"
    # Objective Values:
    #   Processing time: 127
    #   Late jobs: 8
    #   Setup cost: 45
    print(f"Loading instance: {instance_name}")
    problem = load_problem(instance_name, max_nb_batches_per_machine=13)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    print(f"Additional data: {problem.additional_data}")
    result, solver = solve_with_cpsat(problem=problem, time_limit=100, solution=None)
    print(f"Status: {solver.status_solver}")

    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result[-1][0]
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        best_solution.print_summary()
        visualize_solution(best_solution)
    else:
        print("No solution found within time limit.")


if __name__ == "__main__":
    run_cpsat_example()
