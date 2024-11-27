#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from matplotlib import pyplot as plt

from discrete_optimization.generic_rcpsp_tools.solvers.ls import LsGenericRcpspSolver
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsCpsatCallback,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.dp import DpRcpspSolver, dp


def test_basic_stats_callback():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    problem = parse_file(file)
    callback = BasicStatsCallback()
    solver = LsGenericRcpspSolver(problem=problem)
    res = solver.solve(
        callbacks=[callback],
        nb_iteration_max=10000,
        retrieve_intermediate_solutions=True,
    )
    fig, ax = plt.subplots(1)
    ax.plot([x["time"] for x in callback.stats], [x["fit"] for x in callback.stats])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fitness")
    plt.show()
    assert len(callback.stats) > 0


def test_cpsat_callback():
    file = [f for f in get_data_available() if "j1201_5.sm" in f][0]
    problem = parse_file(file)
    callback = StatsCpsatCallback()
    solver = CpSatRcpspSolver(problem=problem)
    res = solver.solve(callbacks=[callback], time_limit=20)
    fig, ax = plt.subplots(1)
    ax.plot(
        [x["time"] for x in callback.stats],
        [x["obj"] for x in callback.stats],
        label="obj function",
    )
    ax.plot(
        [x["time"] for x in callback.stats],
        [x["bound"] for x in callback.stats],
        label="bound",
    )
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Objective")
    plt.show()
    assert len(callback.stats) > 0
    assert all("bound" in x for x in callback.stats)
    assert all("obj" in x for x in callback.stats)
    assert all("time-cpsat" in x for x in callback.stats)
