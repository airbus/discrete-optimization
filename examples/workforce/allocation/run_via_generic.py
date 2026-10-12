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
from discrete_optimization.workforce.allocation.parser import (
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.transformations.to_generic import (
    WorkforceAllocationToGenericSchedulingTransformation,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
)


def run_generic():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    p = ParametersCp.default_cpsat()
    # TODO : reach lowerbound performance of the pure allocation problem ?
    solver = TransformationSolver(
        transformation=WorkforceAllocationToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=allocation_problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol = res[-1][0]
    print(allocation_problem.satisfy(sol), allocation_problem.evaluate(sol))
    assert allocation_problem.satisfy(sol)


if __name__ == "__main__":
    run_generic()
