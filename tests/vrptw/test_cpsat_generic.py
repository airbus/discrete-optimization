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
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.transformations.to_generic import (
    VrpTwToGenericSchedulingTransformation,
)


def test_cpsat_generic_transformation_relaxed_tw():
    """Test VRPTW problem with relaxed time windows."""
    file = [f for f in get_data_available() if "C1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    solver = TransformationSolver(
        transformation=VrpTwToGenericSchedulingTransformation(
            nb_closest=15,
            use_predecessors=False,
            use_first_last=True,
            relax_time_window=True,
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
    if len(res) > 0:
        sol = res[-1][0]
        assert problem.satisfy(sol)
