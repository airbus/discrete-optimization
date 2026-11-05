#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

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
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.transformations.to_generic import (
    TspTwToGenericSchedulingTransformation,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat_generic():
    print(get_data_available())
    problem = parse_tsptw_file(get_data_available()[0])
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 12
    solver = TransformationSolver(
        transformation=TspTwToGenericSchedulingTransformation(
            nb_closest=25,
            use_predecessors=True,
            use_first_last=True,
            relax_time_window=True,
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "logging_objectives": True,
                "parameters_cp": parameters_cp,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    print(problem.time_windows)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_cpsat_generic()
