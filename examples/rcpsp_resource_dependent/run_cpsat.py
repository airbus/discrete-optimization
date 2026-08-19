#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
)
from discrete_optimization.rcpsp_resource_dependent.solvers.cpsat import (
    CpSatRcpspResourceDependentSolver,
)


def create_toy_model():
    resources = {"R1": 5, "R2": 8, "R3": 10, "N1": 20, "N2": 20}
    mode_details = {
        "source": {1: {"duration": 0}},
        "1": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "2": {
            1: {
                "R1": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N1": {frozenset([("1", 1)]): 20, frozenset([("1", 2)]): 10},
                "duration": 2,
            },
            2: {
                "R2": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N2": 5,
                "duration": 1,
            },
        },
        "3": {
            1: {"R1": 1, "N1": 5, "duration": 3},
            2: {"R2": 4, "N2": 5, "duration": 2},
        },
        "4": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 3},
        },
        "5": {
            1: {"R1": 1, "N1": 5, "duration": 4},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "6": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "sink": {1: {"duration": 0}},
    }
    successors = {
        "source": ["1", "2"],
        "1": ["3", "4"],
        "2": ["5"],
        "3": ["4"],
        "4": ["6"],
        "5": ["sink"],
        "6": ["sink"],
        "sink": [],
    }
    problem = RcpspResourceDependentProblem(
        resources=resources,
        non_renewable_resources=["N1", "N2"],
        mode_details=mode_details,
        successors=successors,
        horizon=30,
        source_task="source",
        sink_task="sink",
    )
    solver = CpSatRcpspResourceDependentSolver(problem)
    solver.init_model(avoid_interval_optional=False)
    res = solver.solve(
        time_limit=10, ortools_cpsat_solver_kwargs={"log_search_progress": True}
    )
    sol: RcpspResourceDependentSolution = res[-1][0]
    resource_consumption = {}
    total_conso_nr = {r: 0 for r in problem.non_renewable_resources}
    for t in problem.tasks_list:
        for r in problem.cumulative_resources_list:
            resource_consumption[(t, r)] = sol.get_calendar_resource_consumption(r, t)
        for r in problem.non_renewable_resources_list:
            resource_consumption[(t, r)] = sol.get_non_renewable_resource_consumption(
                r, t
            )
            total_conso_nr[r] += resource_consumption[(t, r)]
    for t in problem.tasks_list:
        for r in (
            problem.cumulative_resources_list + problem.non_renewable_resources_list
        ):
            print(t, r, ":", resource_consumption[(t, r)])
    print(total_conso_nr)
    print(sol.schedule, "\n", sol.modes)
    print(problem.evaluate(sol), problem.satisfy(sol))
    from discrete_optimization.generic_tasks_tools.plot_utils import (
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


if __name__ == "__main__":
    create_toy_model()
