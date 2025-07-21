import matplotlib.pyplot as plt
import networkx as nx

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
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(
        time_limit=5, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    ).get_best_solution()
    print(allocation_problem.evaluate(sol))


def run_lexico():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(
        modelisation_allocation=ModelisationAllocationOrtools.BINARY,
        modelisation_dispersion=ModelisationDispersion.PROXY_MAX_MIN,
    )

    lexico = LexicoSolver(subsolver=solver, problem=allocation_problem)
    from discrete_optimization.generic_tools.callbacks.sequential_solvers_callback import (
        RetrieveSubRes,
    )
    from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
        StatsCpsatCallback,
    )

    retrieve_sub_res = RetrieveSubRes()
    stats_cb = StatsCpsatCallback()
    res = lexico.solve(
        callbacks=[retrieve_sub_res],
        objectives=["nb_teams", "duration"],
        subsolver_callbacks=[stats_cb],
        time_limit=5,
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

    fig, ax = plt.subplots(2)
    fits = [allocation_problem.evaluate(s) for s, _ in res.list_solution_fits]
    ax[0].plot([f["nb_teams"] for f in fits])
    ax[1].plot([f["duration"] for f in fits])
    ax[0].set_title("Nb teams")
    ax[1].set_title("fairness duration")
    plt.show()


if __name__ == "__main__":
    run_lexico()
