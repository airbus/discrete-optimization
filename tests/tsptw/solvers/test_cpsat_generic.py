#  Copyright (c) 2025 AIRBUS and its affiliates.
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
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.transformations.to_generic import (
    TspTwToGenericSchedulingTransformation,
)


def test_cpsat_generic_transformation_relaxed_tw():
    """Test TSPTW problem with relaxed time windows."""
    problem = parse_tsptw_file(get_data_available()[0])

    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 4

    solver = TransformationSolver(
        transformation=TspTwToGenericSchedulingTransformation(
            nb_closest=20,
            use_predecessors=False,
            use_first_last=False,
            relax_time_window=True,
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": parameters_cp,
            },
        ),
        source_problem=problem,
    )

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    if len(res) > 0:
        sol = res[-1][0]
        # Note: with relaxed time windows, solution might not strictly satisfy original TW constraints
        print(f"Evaluation: {problem.evaluate(sol)}")
