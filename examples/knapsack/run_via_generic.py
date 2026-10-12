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
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.transformations.to_generic import (
    KnapsackToGenericSchedulingTransformation,
)


def run_via_generic_knapsack():
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    p = ParametersCp.default_cpsat()
    solver = TransformationSolver(
        transformation=KnapsackToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=knapsack_problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol = res[-1][0]
    print(knapsack_problem.satisfy(sol), knapsack_problem.evaluate(sol))
    assert knapsack_problem.satisfy(sol)


if __name__ == "__main__":
    run_via_generic_knapsack()
