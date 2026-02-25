#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.optal import (
    OptalRcpspSolver,
    optalcp_available,
)


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
@pytest.mark.parametrize("filename", ["j1201_7.sm", "j1010_1.mm"])
def test_optal(filename):
    files_available = get_data_available()
    file = [f for f in files_available if filename in f][0]
    problem = parse_file(file)
    solver = OptalRcpspSolver(problem=problem)
    solver.init_model()
    stats_cb = StatsWithBoundsCallback()
    res = solver.solve(
        callbacks=[stats_cb],
        time_limit=2,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
    df = stats_cb.get_df_metrics()
    assert len(df) >= 2
    last_line = df.iloc[-1]
    assert last_line["obj"] == solver.current_obj
    assert last_line["bound"] == solver.current_bound
