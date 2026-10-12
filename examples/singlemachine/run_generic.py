#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.solvers.cpsat import WTSolution
from discrete_optimization.singlemachine.transformations.to_generic import (
    SingleMachineToGenericSchedulingTransformation,
)

logging.basicConfig(level=logging.INFO)


def run_generic():
    problem = parse_file(get_data_available()[0])[0]
    p = ParametersCp.default_cpsat()
    # TODO investigate the difference of perfo?
    solver = TransformationSolver(
        transformation=SingleMachineToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            kwargs={
                "parameters_cp": p,
                "time_limit": 20,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    solver.init_model()
    res = solver.solve()
    sol: WTSolution = res.get_best_solution()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_generic()
