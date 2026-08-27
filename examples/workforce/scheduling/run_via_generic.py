#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import plotly.io as pio

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
    Objective,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    AllocSchedulingProblem,
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_auto import (
    CPSatAutoAllocSchedulingSolver,
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.transformations.generic_scheduling_impl import (
    WfSchedulingToGenericSchedulingTransformation,
)
from discrete_optimization.workforce.scheduling.utils import (
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_191.json" in p][0]
    problem = parse_json_to_problem(instance)
    # problem.same_allocation = []
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    obj_computer = [
        obj
        for obj in problem.get_list_objective_computer()
        if obj.get_objective_name()
        in [Objective.NB_UNARY_RESOURCES_USED, Objective.CUMUL_COST]
    ]
    print(obj_computer)
    solver = TransformationSolver(
        transformation=WfSchedulingToGenericSchedulingTransformation(obj_computer),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "exactly_one_unary_resource_per_task": True,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)


def run_cpsat_disrupted():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = CPSatAutoAllocSchedulingSolver(problem)
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
    new_problem: AllocSchedulingProblem = d["scheduling_problem"]
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    from discrete_optimization.generic_tasks_tools.objectives.allocated_tasks import (
        AllocatedTasksObjective,
    )
    from discrete_optimization.generic_tasks_tools.objectives.allocation_changes import (
        AllocationSwitchObjectiveComputer,
    )
    from discrete_optimization.generic_tasks_tools.objectives.schedule_changes import (
        ScheduleChangesComputer,
    )
    from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
        UnaryResourcesUsedComputer,
    )

    obj_computer = [
        AllocationSwitchObjectiveComputer(
            problem=new_problem,
            base_allocation_solution=sol,
            weight_objective=1.0,
            switch_on_cost={
                t: {ur: 100 for ur in new_problem.unary_resources_list}
                for t in new_problem.tasks_list
            },
            switch_off_cost={
                t: {ur: 100 for ur in new_problem.unary_resources_list}
                for t in new_problem.tasks_list
            },
        ),
        ScheduleChangesComputer(
            problem=new_problem,
            base_scheduling_solution=sol,
            weight_objective=1,
            cost_any_shift={t: 1 for t in new_problem.tasks_list},
            cost_unit_deviation={t: 1 for t in new_problem.tasks_list},
        ),
        AllocatedTasksObjective(problem=new_problem, weight_objective=-1000),
        UnaryResourcesUsedComputer(
            problem,
            weight_objective=1,
            weight_per_unary_resource={
                ur: 1 for ur in new_problem.unary_resources_list
            },
        ),
    ]
    print(obj_computer)
    solver = TransformationSolver(
        transformation=WfSchedulingToGenericSchedulingTransformation(obj_computer),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 100,
                "exactly_one_unary_resource_per_task": True,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=new_problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    new_sol = res[-1][0]
    print(
        "Objectives :",
        obj_computer[0].compute_objective(new_sol),
        obj_computer[1].compute_objective(new_sol),
    )
    plotly_schedule_comparison(
        base_solution=sol,
        updated_solution=new_sol,
        show_all_changes=True,
        problem=d["scheduling_problem"],
        use_color_map_per_task=False,
        color_map_per_task={},
        plot_team_breaks=True,
        display=True,
    )


if __name__ == "__main__":
    run_cpsat_disrupted()
