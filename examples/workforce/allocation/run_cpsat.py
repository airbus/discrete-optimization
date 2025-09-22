#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import networkx as nx

from discrete_optimization.generic_tools.callbacks.sequential_solvers_callback import (
    RetrieveSubRes,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsCpsatCallback,
)
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
    ModelisationDispersion,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(
        time_limit=5, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    ).get_best_solution()
    print(allocation_problem.evaluate(sol))


def run_lexico():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(
        modelisation_allocation=ModelisationAllocationOrtools.BINARY,
        modelisation_dispersion=ModelisationDispersion.EXACT_MODELING_WITH_IMPLICATION,
    )
    lexico = LexicoSolver(subsolver=solver, problem=allocation_problem)

    retrieve_sub_res = RetrieveSubRes()
    stats_cb = StatsCpsatCallback()
    res = lexico.solve(
        callbacks=[retrieve_sub_res],
        objectives=["nb_teams", "duration"],
        subsolver_callbacks=[stats_cb],
        time_limit=5,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    fig, ax = plt.subplots(2)
    fits = [
        [allocation_problem.evaluate(s) for s in res_]
        for res_ in retrieve_sub_res.sol_per_step
    ]
    ax[0].plot([f["nb_teams"] for f in fits[0]])
    ax[1].plot([f["duration"] for f in fits[1]])
    ax[0].set_title("Phase optim Nb teams")
    ax[1].set_title("Phase optim fairness duration")
    print(fits)

    fig, ax = plt.subplots(2)
    fits = [allocation_problem.evaluate(s) for s, _ in res.list_solution_fits]
    ax[0].plot([f["nb_teams"] for f in fits])
    ax[1].plot([f["duration"] for f in fits])
    ax[0].set_title("Nb teams")
    ax[1].set_title("fairness duration")
    plt.show()


if __name__ == "__main__":
    run_cpsat()
