from datetime import datetime

import networkx as nx
import plotly.io as pio

from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
)
from discrete_optimization.workforce.allocation.problem import satisfy_same_allocation
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

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.


def run_disruption_creation():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    fig = plot_allocation_solution(
        problem=allocation_problem,
        sol=sol,
        display=False,
        ref_date=datetime(year=2024, month=1, day=1),
    )
    fig.show()
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

    for d in ds:
        new_alloc = d["new_allocation_problem"]
        plot_allocation_solution(
            problem=new_alloc,
            sol=d["new_solution"],
            use_color_map=True,
            plot_breaks=True,
        )
        print("---")
        print(new_alloc.calendar_team)
        print(new_alloc.satisfy(d["new_solution"]))

        print(new_alloc.evaluate(d["new_solution"]))
        print("---")


if __name__ == "__main__":
    run_disruption_creation()
