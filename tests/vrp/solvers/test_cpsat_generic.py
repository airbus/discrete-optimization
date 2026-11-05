#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.transformations.to_generic import (
    VrpToGenericSchedulingTransformation,
)


def test_cpsat_generic_transformation():
    """Test VRP problem solved via generic scheduling transformation."""
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]

    p = ParametersCp.default_cpsat()
    p.nb_process = 4  # Reduce for testing

    solver = TransformationSolver(
        transformation=VrpToGenericSchedulingTransformation(
            nb_closest=20, use_predecessors=True, use_first_last=False
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": p,
            },
        ),
        source_problem=problem,
    )

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    sol = res[-1][0]
    assert problem.satisfy(sol)
    print(f"Solution paths: {sol.list_paths}")
    print(f"Evaluation: {problem.evaluate(sol)}")


def test_cpsat_generic_transformation_with_first_last():
    """Test VRP problem with first/last entity constraints."""
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = TransformationSolver(
        transformation=VrpToGenericSchedulingTransformation(
            nb_closest=15, use_predecessors=False, use_first_last=True
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": p,
            },
        ),
        source_problem=problem,
    )

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    sol = res[-1][0]
    assert problem.satisfy(sol)
