#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from functools import partial

import plotly.io as pio
from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
    ObjectivesEnum,
    _get_variables_obj_key,
)
from discrete_optimization.workforce.scheduling.utils import (
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat_lexico_delta_objective():
    # Load the original scheduling problem
    files = get_data_available()
    file_path = files[4]
    problem_original = parse_json_to_problem(file_path)
    solver_original = CPSatAllocSchedulingSolver(problem_original)
    solver_original.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS],
        adding_redundant_cumulative=True,
        add_lower_bound=True,
    )
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
            lower_nb_disruption=1,
            upper_nb_disruption=1,
            lower_nb_teams=1,
            upper_nb_teams=1,
            duration_discrete_distribution=(
                [200],
                [1],
            ),
        ),
    )
    problem_disrupted = disruption_scenario["scheduling_problem"]
    additional_constraints = disruption_scenario["additional_constraint_scheduling"]
    print("Disrupted problem created.")

    # 3. Plot the original, now-infeasible plan for reference
    print("\nOriginal plan (now likely infeasible due to disruption):")
    # plotly_schedule_comparison(
    #     sol_original,
    #     sol_original,
    #     problem_disrupted,
    #     title="Original Plan (Now Infeasible)",
    #     display=True,
    #     plot_team_breaks=True,
    # )

    # Define the objective order: first reallocations, then shifts.
    objectives_realloc_first = [
        ObjectivesEnum.NB_DONE_AC,
        "reallocated",
        "nb_shifted",
        ObjectivesEnum.DISPERSION,
        "sum_delta_schedule",
        # "max_delta_schedule",
        # ObjectivesEnum.NB_TEAMS,
    ]

    # Set up the solver with all possible similarity objectives
    solver_realloc = CPSatAllocSchedulingSolver(problem_disrupted)
    solver_realloc.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.DISPERSION,
            ObjectivesEnum.NB_TEAMS,
        ],
        additional_constraints=additional_constraints,
        optional_activities=True,
        base_solution=sol_original,
    )

    def get_val(sol: Solution, obj):
        return sol._intern_obj[obj]

    dict_function = {obj: partial(get_val, obj=obj) for obj in objectives_realloc_first}
    pareto = CpsatParetoSolver(
        problem=problem_disrupted,
        solver=solver_realloc,
        objective_names=objectives_realloc_first,
        dict_function=dict_function,
        # Idea :
        # Nadir points,
        # calculer les points extreme pour initialiser les discretisation.
        delta_abs_improvement=[2] * len(objectives_realloc_first),
        delta_ref_improvement=[0.05] * len(objectives_realloc_first),
    )
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    front = pareto.solve(
        [
            solver_realloc.variables["objectives"][_get_variables_obj_key(obj)]
            for obj in objectives_realloc_first
        ],
        time_limit=120,
        subsolver_kwargs={
            "time_limit": 4,
            "ortools_cpsat_solver_kwargs": {"log_search_progress": False},
            "parameters_cp": p,
        },
    )
    print(f"\nFound {len(front)} Pareto solutions:")
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        print(sol._intern_obj)
        f1s.append(pareto.dict_function["reallocated"](sol))
        f2s.append(pareto.dict_function["nb_shifted"](sol))
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("realloc")
    plt.ylabel("nbshifted")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pareto_rcpsp.png")
    for sol, _ in front:
        plotly_schedule_comparison(
            sol_original,
            sol,
            problem_disrupted,
            title="Original Plan vs sol",
            display=True,
            plot_team_breaks=True,
        )
    plt.show()


if __name__ == "__main__":
    run_cpsat_lexico_delta_objective()
