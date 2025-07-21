#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

import plotly.io as pio
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.sequential_solvers_callback import (
    RetrieveSubRes,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsCpsatCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    create_scheduling_problem_several_resource_dropping,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_relaxed import (
    CPSatAllocSchedulingSolverCumulative,
)
from discrete_optimization.workforce.scheduling.utils import plotly_schedule_comparison

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_cpsat_delta():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    sol, _ = solver.solve(time_limit=2).get_best_solution_fit()
    d = generate_scheduling_disruption(
        original_scheduling_problem=problem,
        original_solution=sol,
        list_drop_resource=None,
        params_randomness=ParamsRandomness(
            lower_nb_disruption=1,
            upper_nb_disruption=2,
            lower_nb_teams=1,
            upper_nb_teams=1,
        ),
    )
    solver_relaxed = CPSatAllocSchedulingSolver(d["scheduling_problem"])
    solver_relaxed.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
        ],
        additional_constraints=d["additional_constraint_scheduling"],
        optional_activities=True,
        base_solution=sol,
    )
    res = solver_relaxed.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    changes = compute_changes_between_solution(solution_a=sol, solution_b=res[-1][0])
    for key in ["nb_reallocated", "nb_shift", "mean_shift", "sum_shift", "max_shift"]:
        print(key, " : ", changes[key])
    plotly_schedule_comparison(
        base_solution=sol,
        updated_solution=res[-1][0],
        problem=d["scheduling_problem"],
        use_color_map_per_task=False,
        color_map_per_task={},
        plot_team_breaks=True,
        display=True,
    )
    print(d)


def run_cpsat_lexico():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.DISPERSION,
            ObjectivesEnum.MAKESPAN,
        ],
        adding_redundant_cumulative=True,
    )
    lexico_solver = LexicoSolver(subsolver=solver, problem=problem)
    retrieve_sub_res = RetrieveSubRes()
    res = lexico_solver.solve(
        callbacks=[retrieve_sub_res],
        objectives=[
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.DISPERSION,
            ObjectivesEnum.MAKESPAN,
        ],
        time_limit=5,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    fig, ax = plt.subplots(2)
    fits = [
        [problem.evaluate(s) for s in res_] for res_ in retrieve_sub_res.sol_per_step
    ]
    ax[0].plot([f["nb_teams"] for f in fits[0]])
    ax[1].plot([f["workload_dispersion"] for f in fits[1]])
    ax[0].set_title("Phase optim Nb teams")
    ax[1].set_title("Phase optim fairness duration")

    fig, ax = plt.subplots(2)
    fits = [problem.evaluate(s) for s, _ in res.list_solution_fits]
    ax[0].plot([f["nb_teams"] for f in fits])
    ax[1].plot([f["workload_dispersion"] for f in fits])
    ax[0].set_title("Nb teams")
    ax[1].set_title("fairness duration")
    plt.show()


def run_cpsat_lexico_delta():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    sol, _ = solver.solve(time_limit=2).get_best_solution_fit()
    d = generate_scheduling_disruption(
        original_scheduling_problem=problem,
        original_solution=sol,
        list_drop_resource=None,
        params_randomness=ParamsRandomness(
            lower_nb_disruption=1,
            upper_nb_disruption=2,
            lower_nb_teams=1,
            upper_nb_teams=1,
        ),
    )
    solver_relaxed = CPSatAllocSchedulingSolver(d["scheduling_problem"])
    solver_relaxed.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.DISPERSION,
            ObjectivesEnum.MAKESPAN,
        ],
        additional_constraints=d["additional_constraint_scheduling"],
        optional_activities=True,
        base_solution=sol,
    )
    retrieve_sub_res = RetrieveSubRes()
    lexico_solver = LexicoSolver(
        subsolver=solver_relaxed,
        problem=d["scheduling_problem"],
    )
    res = lexico_solver.solve(
        callbacks=[retrieve_sub_res],
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.DISPERSION,
            ObjectivesEnum.MAKESPAN,
        ],
        time_limit=5,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    fits = [
        [problem.evaluate(s) for s in res_] for res_ in retrieve_sub_res.sol_per_step
    ]
    for j in range(len(retrieve_sub_res.sol_per_step)):
        res_ = retrieve_sub_res.sol_per_step[j]
        for k in range(len(res_)):
            sol_ = res_[k]
            changes = compute_changes_between_solution(solution_a=sol, solution_b=sol_)
            for key in [
                "nb_reallocated",
                "nb_shift",
                "mean_shift",
                "sum_shift",
                "max_shift",
            ]:
                fits[j][k][key] = changes[key]
            fits[j][k]["changes"] = sum(
                [
                    fits[j][k][key]
                    for key in [
                        "nb_reallocated",
                        "nb_shift",
                        "mean_shift",
                        "sum_shift",
                        "max_shift",
                    ]
                ]
            )
    fig, ax = plt.subplots(5)
    ax[0].plot([f["nb_not_done"] for f in fits[0]])
    ax[1].plot([f["changes"] for f in fits[1]])
    ax[2].plot([f["nb_teams"] for f in fits[2]])
    ax[3].plot([f["workload_dispersion"] for f in fits[3]])
    ax[4].plot([f["makespan"] for f in fits[4]])

    ax[0].set_title("Phase optim Nb task done")
    ax[1].set_title("Phase optim Changes to base solution")
    ax[2].set_title("Phase optim Nb teams")
    ax[3].set_title("Phase optim Dispersion workload")
    ax[4].set_title("Phase optim makespan")
    plt.show()


if __name__ == "__main__":
    run_cpsat_lexico_delta()
