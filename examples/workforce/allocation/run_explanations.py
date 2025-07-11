from datetime import datetime

import networkx as nx
import plotly.io as pio

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpyCorrectUnsatMethod,
    CpmpyExplainUnsatMethod,
)
from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.allocation.problem import satisfy_same_allocation
from discrete_optimization.workforce.allocation.solvers.cpmpy import (
    CPMpyTeamAllocationSolver,
    CPMpyTeamAllocationSolverStoreConstraintInfo,
    ModelisationAllocationCP,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.allocation.utils import plot_allocation_solution
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    create_scheduling_problem_several_resource_dropping,
    cut_number_of_team,
    generate_allocation_disruption,
    generate_resource_disruption_scenario_from,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.utils import (
    alloc_solution_to_alloc_sched_solution,
    build_scheduling_problem_from_allocation,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.


def run_disruption_creation():
    instances = [p for p in get_data_available()]
    scheduling_problem = parse_json_to_problem(instances[1])
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    nb_teams = allocation_problem.evaluate(sol)["nb_teams"]
    # fig = plot_allocation_solution(problem=allocation_problem, sol=sol, display=False,
    #                                ref_date=datetime(year=2024, month=1, day=1))
    # fig.show()
    ds = [
        generate_allocation_disruption(
            original_allocation_problem=allocation_problem,
            original_solution=sol,
            params_randomness=ParamsRandomness(
                lower_nb_disruption=1,
                upper_nb_disruption=1,
                lower_nb_teams=1,
                upper_nb_teams=2,
                lower_time=0,
                upper_time=600,
                duration_discrete_distribution=(
                    [15, 30, 60, 120],
                    [0.25, 0.25, 0.25, 0.25],
                ),
            ),
        )
        for i in range(10)
    ]
    # problem = ds[0]["new_allocation_problem"]
    allocation_problem.allocation_additional_constraint.nb_max_teams = 9
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem=allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationCP.BINARY)
    res = solver.solve(time_limit=2)
    print(solver.status_solver)

    explanations = solver.explain_unsat_meta(
        soft=solver.meta_constraints,
        hard=[],
        cpmpy_method=CpmpyExplainUnsatMethod.mus,
        solver="exact",
    )
    sched_problem = build_scheduling_problem_from_allocation(problem=allocation_problem)
    sched_scheduling = alloc_solution_to_alloc_sched_solution(
        problem=sched_problem, solution=sol
    )
    tasks_in_conflict = []
    for mc in explanations:
        if mc.metadata["type"] == "allocated_task":
            print(mc.metadata)
            tasks_in_conflict.append(mc.metadata["task_index"])
    plotly_schedule_comparison(
        base_solution=sched_scheduling,
        updated_solution=sched_scheduling,
        problem=sched_problem,
        use_color_map_per_task=True,
        color_map_per_task={i: "red" for i in tasks_in_conflict},
        display=True,
    )
    for exp in explanations:
        print(exp)


if __name__ == "__main__":
    run_disruption_creation()
