#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.plot import plot_vrp_solution
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.transformations.to_generic import (
    VrpToGenericSchedulingTransformation,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat_generic_solver():
    file = [f for f in get_data_available() if "vrp_76_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=VrpToGenericSchedulingTransformation(
            nb_closest=20, use_predecessors=True, use_first_last=False
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    subsolver: GenericSchedulingAutoCpSatImplSolver = solver.wrapped_solver
    for i_routing in subsolver.next_variable_routing:
        for i, j, var in subsolver.next_variable_routing[i_routing]:
            value = subsolver.solver.value(var)
            if value == 1:
                print(i_routing, i, j, value)
                if i != "depot":
                    print(
                        "is active",
                        subsolver.solver.value(
                            subsolver.get_entity_is_active_variable(i)
                        ),
                    )
                if j != "depot":
                    print(
                        "is active",
                        subsolver.solver.value(
                            subsolver.get_entity_is_active_variable(j)
                        ),
                    )
    print(solver.status_solver)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    print(sol.list_paths)
    plot_vrp_solution(vrp_problem=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    run_cpsat_generic_solver()
