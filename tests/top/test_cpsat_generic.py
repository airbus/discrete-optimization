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
from discrete_optimization.top.parser import get_data_available, parse_file
from discrete_optimization.top.transformations.to_generic import (
    TopToGenericSchedulingTransformation,
)


def test_cpsat_generic_transformation():
    """Test TOP problem solved via generic scheduling transformation."""
    files, files_dict = get_data_available()
    file = [f for f in files if "p2.2.a.txt" in f][0]
    problem = parse_file(file)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4  # Reduce for testing

    solver = TransformationSolver(
        transformation=TopToGenericSchedulingTransformation(
            nb_closest=20, use_predecessors=False, use_first_last=True
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


def test_cpsat_generic_transformation_with_predecessors():
    """Test TOP problem with predecessors constraint."""
    files, files_dict = get_data_available()
    file = [f for f in files if "p2.2.a.txt" in f][0]
    problem = parse_file(file)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = TransformationSolver(
        transformation=TopToGenericSchedulingTransformation(
            nb_closest=15, use_predecessors=True, use_first_last=False
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
