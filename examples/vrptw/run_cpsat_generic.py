#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
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
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.transformations.to_generic import (
    VrpTwToGenericSchedulingTransformation,
)


def run_cpsat_generic():
    file = [f for f in get_data_available() if "C1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=VrpTwToGenericSchedulingTransformation(
            nb_closest=20,
            use_predecessors=True,
            use_first_last=False,
            relax_time_window=True,
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
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat_generic()
