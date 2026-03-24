import optalcp as cp

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcalbp_l.parser import parse_rcalbpl_json
from discrete_optimization.rcalbp_l.solvers.optal import (
    OptalRCALBPLSolver,
)
from examples.rcalbp_l.plot import plot_rcalbpl_dashboard


def main():
    problem = parse_rcalbpl_json("instances/187_2_26_2880.json")
    problem.nb_periods = 5
    problem.periods = range(problem.nb_periods)
    solver = OptalRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()

    p.nb_process = 12
    res = solver.solve(
        time_limit=100,
        parameters_cp=p,
        workers=[
            cp.WorkerParameters(
                searchType="FDS", noOverlapPropagationLevel=4, cumulPropagationLevel=3
            ),
            cp.WorkerParameters(
                searchType="FDSDual",
                noOverlapPropagationLevel=4,
                cumulPropagationLevel=3,
            ),
        ]
        * 2,
    )
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


def main_easy():
    problem = parse_rcalbpl_json("instances/187_2_26_2880.json")
    print(problem.durations)
    problem.durations = [
        [problem.durations[t][-1]] * len(problem.durations[t])
        for t in range(problem.nb_tasks)
    ]
    problem.nb_periods = 10
    problem.periods = range(problem.nb_periods)
    solver = OptalRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    res = solver.solve(
        time_limit=20,
        parameters_cp=p,
        workers=[
            cp.WorkerParameters(searchType="FDS"),
            cp.WorkerParameters(searchType="FDSDual"),
            cp.WorkerParameters(searchType="LNS"),
            cp.WorkerParameters(searchType="LNS"),
        ],
    )
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    main()
