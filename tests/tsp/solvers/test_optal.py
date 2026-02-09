#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pandas as pd

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.solvers.optal import OptalTspSolver


def test_optal():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    problem = parse_file(files[0], start_index=0, end_index=0)
    solver = OptalTspSolver(problem=problem)
    solver.init_model(scaling=1)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        do_not_retrieve_solutions=True,  # free license = no solutions stored
        **{
            "worker0-1.searchType": "fdslb",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-1.cumulPropagationLevel": 3,
        },
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
