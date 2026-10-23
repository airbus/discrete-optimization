#  Copyright (c) 2026 AIRBUS and its affiliates.
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
from discrete_optimization.top.parser import get_data_available, parse_file
from discrete_optimization.top.transformations.to_generic import (
    TopToGenericSchedulingTransformation,
)
from discrete_optimization.vrp.problem import VrpSolution

logging.basicConfig(level=logging.INFO)


def run():
    files, files_dict = get_data_available()
    print(files)
    file = [f for f in files if "pr299_gen1_m3.txt" in f][0]
    problem = parse_file(file)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=TopToGenericSchedulingTransformation(
            nb_closest=20, use_predecessors=False, use_first_last=True
        ),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "logging_objectives": True,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol: VrpSolution = res[-1][0]
    print(sol.list_paths)
    print(problem.evaluate_function(sol))
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run()
