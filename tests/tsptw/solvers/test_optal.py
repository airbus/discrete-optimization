#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pandas as pd

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.tsptw.solvers.optal import OptalTspTwSolver


def test_optal(problem):
    solver = OptalTspTwSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        time_limit=2,
        do_not_retrieve_solutions=True,  # free license = no solutions stored
    )
    assert len(res) == 0
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    stats = solver.get_output_stats()
    stats_df = pd.concat(
        (
            pd.DataFrame(stats["objectiveHistory"]).set_index("solveTime"),
            pd.DataFrame(stats["lowerBoundHistory"]).set_index("solveTime"),
        )
    )
    assert not stats_df.empty
