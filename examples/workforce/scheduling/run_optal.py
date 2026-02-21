#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import optalcp as cp
import plotly.io as pio

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers import ObjectivesEnum
from discrete_optimization.workforce.scheduling.solvers.optal import (
    OptalAllocSchedulingSolver,
)
from discrete_optimization.workforce.scheduling.utils import (
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_optal():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = OptalAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.MAKESPAN,
            # ObjectivesEnum.DISPERSION
        ],
        add_lower_bound=False,
        symmbreak_on_used=True,
        adding_redundant_cumulative=True,
    )
    import optalcp as cp

    workers = [
        cp.WorkerParameters(
            searchType="FDS", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
        cp.WorkerParameters(
            searchType="FDSDual", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
        cp.WorkerParameters(
            searchType="SetTimes", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
    ]
    params = ParametersCp.default_cpsat()
    params.nb_process = 20
    res = solver.solve(
        parameters_cp=params,
        time_limit=30,
        workers=workers,
        searchType="LNS",
        relativeGapTolerance=0.00001,
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    print(solver.status_solver)
    plotly_schedule_comparison(sol, sol, problem, display=True)


def run_optal_lexico():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = OptalAllocSchedulingSolver(problem)
    solver.init_model(objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION])
    import optalcp as cp

    workers = [
        cp.WorkerParameters(
            searchType="FDS", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
        cp.WorkerParameters(
            searchType="FDSDual", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
    ]
    params = ParametersCp.default_cpsat()
    params.nb_process = 10
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver

    lexico_solver = LexicoSolver(subsolver=solver, problem=problem)
    res = lexico_solver.solve(
        objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION],
        parameters_cp=ParametersCp.default_cpsat(),
        workers=workers,
        time_limit=10,
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)


def run_optal_delta():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = OptalAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    sol, _ = solver.solve(
        time_limit=2, parameters_cp=ParametersCp.default_cpsat()
    ).get_best_solution_fit()
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
    solver_relaxed = OptalAllocSchedulingSolver(d["scheduling_problem"])
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
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver_relaxed.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=30,
        relativeGapTolerance=0.0000,
        workers=[
            cp.WorkerParameters(
                searchType="FDS", cumulPropagationLevel=3, noOverlapPropagationLevel=4
            ),
            cp.WorkerParameters(
                searchType="FDSDual",
                cumulPropagationLevel=3,
                noOverlapPropagationLevel=4,
            ),
        ],
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    changes = compute_changes_between_solution(solution_a=sol, solution_b=res[-1][0])
    for key in ["nb_reallocated", "nb_shift", "mean_shift", "sum_shift", "max_shift"]:
        print(key, " : ", changes[key])
    print("satisfy = ", res[-1][0].problem.satisfy(res[-1][0]))
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


def run_optal_delta_lexico():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = OptalAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    sol, _ = solver.solve(
        time_limit=2, parameters_cp=ParametersCp.default_cpsat()
    ).get_best_solution_fit()
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
    solver_relaxed = OptalAllocSchedulingSolver(d["scheduling_problem"])
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
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    workers = [
        cp.WorkerParameters(
            searchType="FDS", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
        cp.WorkerParameters(
            searchType="FDSDual", cumulPropagationLevel=3, noOverlapPropagationLevel=4
        ),
    ]
    lexico_solver = LexicoSolver(subsolver=solver_relaxed, problem=problem)
    res = lexico_solver.solve(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            "reallocated",
            "sum_delta_schedule",
            "max_delta_schedule",
            "nb_shifted",
        ],
        parameters_cp=p,
        workers=workers,
        time_limit=10,
        relativeGapTolerance=0.0000,
    )

    changes = compute_changes_between_solution(solution_a=sol, solution_b=res[-1][0])
    for key in ["nb_reallocated", "nb_shift", "mean_shift", "sum_shift", "max_shift"]:
        print(key, " : ", changes[key])
    print("satisfy = ", res[-1][0].problem.satisfy(res[-1][0]))
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


if __name__ == "__main__":
    run_optal_delta_lexico()
