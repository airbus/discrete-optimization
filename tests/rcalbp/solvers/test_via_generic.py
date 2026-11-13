#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.alb.rcalbp.problem import RCALBPSolution
from discrete_optimization.alb.rcalbp.transformations.to_generic import (
    RCALBPToGenericSchedulingTransformation,
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


def test_via_generic(
    simple_problem,
    generic_solver_cls,
    generic_solver_kwargs,
    generic_solver_check_solution,
):
    # Load problem from RCPSP instance
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=RCALBPToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            generic_solver_cls,
            {
                "time_limit": 100,
                "parameters_cp": p,
                **generic_solver_kwargs,
            },
        ),
        source_problem=simple_problem,
    )

    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    if generic_solver_check_solution:
        assert len(res) > 0
        solution: RCALBPSolution = res.get_best_solution()
        assert simple_problem.satisfy(solution)
