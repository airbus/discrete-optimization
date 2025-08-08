#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Optional

import plotly.io as pio
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.sequential_solvers_callback import (
    RetrieveSubRes,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsCpsatCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
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
from discrete_optimization.workforce.scheduling.utils import (
    compute_changes_between_solution,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_196.json" in p][0]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
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
    plotly_schedule_comparison(sol, sol, problem, display=True)


def run_cpsat_delta():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
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
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.DISPERSION,
            # ObjectivesEnum.MAKESPAN,
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
            # ObjectivesEnum.MAKESPAN,
        ],
        time_limit=10,
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


def run_cpsat_lexico_delta_objective():
    # Load the original scheduling problem
    files = get_data_available()
    file_path = files[1]
    problem_original = parse_json_to_problem(file_path)
    solver_original = CPSatAllocSchedulingSolver(problem_original)
    solver_original.init_model(objectives=[ObjectivesEnum.NB_TEAMS])
    result_original = solver_original.solve(
        time_limit=5, parameters_cp=ParametersCp.default_cpsat()
    )
    sol_original, fit_original = result_original.get_best_solution_fit()
    fits = problem_original.evaluate(sol_original)
    print(f"Original optimal solution found with {fits['nb_teams']} teams.")
    disruption_scenario = generate_scheduling_disruption(
        original_scheduling_problem=problem_original,
        original_solution=sol_original,
        params_randomness=ParamsRandomness(
            upper_nb_disruption=1,
            lower_nb_teams=1,
            upper_nb_teams=3,
            duration_discrete_distribution=(
                [60],
                [1],
            ),
        ),
    )
    problem_disrupted = disruption_scenario["scheduling_problem"]
    additional_constraints = disruption_scenario["additional_constraint_scheduling"]
    print("Disrupted problem created.")

    # 3. Plot the original, now-infeasible plan for reference
    print("\nOriginal plan (now likely infeasible due to disruption):")
    plotly_schedule_comparison(
        sol_original,
        sol_original,
        problem_disrupted,
        title="Original Plan (Now Infeasible)",
        display=True,
        plot_team_breaks=True,
    )

    # Define the objective order: first reallocations, then shifts.
    objectives_realloc_first = [
        ObjectivesEnum.NB_DONE_AC,
        "reallocated",
        "nb_shifted",
        "sum_delta_schedule",
        "max_delta_schedule",
        ObjectivesEnum.NB_TEAMS,
    ]

    # Set up the solver with all possible similarity objectives
    solver_realloc = CPSatAllocSchedulingSolver(problem_disrupted)
    solver_realloc.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
        ],
        additional_constraints=additional_constraints,
        optional_activities=True,
        base_solution=sol_original,
    )

    class LexicoCpsatPrevStartCallback(Callback):
        def on_step_end(
            self, step: int, res: ResultStorage, solver: LexicoSolver
        ) -> Optional[bool]:
            subsolver: CPSatAllocSchedulingSolver = solver.subsolver
            subsolver.set_warm_start_from_previous_run()

    lexico_solver = LexicoSolver(subsolver=solver_realloc, problem=problem_disrupted)
    retrieve_sub_res = RetrieveSubRes()
    lexico_cpsat_warm_start = LexicoCpsatPrevStartCallback()
    cbs = [retrieve_sub_res, lexico_cpsat_warm_start]
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 16
    result_realloc = lexico_solver.solve(
        callbacks=cbs,
        parameters_cp=parameters_cp,
        objectives=objectives_realloc_first,
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol_realloc_first = result_realloc[-1][0]

    # Analyze and display the result
    changes_realloc = compute_changes_between_solution(sol_original, sol_realloc_first)
    print("--- Strategy A: Reallocation First ---")
    for key in ["nb_reallocated", "nb_shift", "sum_shift", "max_shift"]:
        print(f"{key}: {changes_realloc[key]}")
    plotly_schedule_comparison(
        sol_original,
        sol_realloc_first,
        problem_disrupted,
        title="Repaired Plan: Prioritizing Fewest Reallocations",
        display=True,
        plot_team_breaks=True,
    )


if __name__ == "__main__":
    run_cpsat_lexico_delta_objective()
