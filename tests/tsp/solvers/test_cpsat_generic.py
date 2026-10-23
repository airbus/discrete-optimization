#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
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
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.transformations.to_generic import (
    TspToGenericSchedulingTransformation,
)


def test_cpsat_generic_transformation():
    """Test TSP problem solved via generic scheduling transformation."""
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4  # Reduce for testing

    solver = TransformationSolver(
        transformation=TspToGenericSchedulingTransformation(
            nb_closest=10, use_predecessors=False, use_first_last=True
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": p,
            },
        ),
        source_problem=model,
    )

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    sol = res[-1][0]
    assert model.satisfy(sol)
    print(f"Evaluation: {model.evaluate(sol)}")


def test_cpsat_generic_transformation_with_predecessors():
    """Test TSP problem with predecessors constraint."""
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = TransformationSolver(
        transformation=TspToGenericSchedulingTransformation(
            nb_closest=15, use_predecessors=True, use_first_last=False
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": p,
            },
        ),
        source_problem=model,
    )

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    sol = res[-1][0]
    assert model.satisfy(sol)
