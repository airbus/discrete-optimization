#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random

from discrete_optimization.generic_tasks_tools.entities import (
    ConstantDurationEntity,
    GroupEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.plot_utils import (
    plot_ressource_view,
    plot_task_gantt,
    plt,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
    SpanBlockingConstraint,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)


def run_constant_duration_entity_example():
    entities = []
    duration_per_mode = {}
    successors = {}
    for product in range(5):
        keys_for_product = set()
        for nb_task in range(8):
            name_task = f"task-prod{product}-{nb_task}"
            duration_per_mode[name_task] = {0: random.randint(2, 5)}
            keys_for_product.add(name_task)
            if nb_task > 0:
                successors[f"task-prod{product}-{nb_task - 1}"] = [name_task]
        entities.append(
            ConstantDurationEntity(
                other_entity=GroupEntity(frozenset(keys_for_product)),
                constant_duration=10,
                start_or_end=StartOrEnd.START,
            )
        )
    span_blocking_constraints = [
        SpanBlockingConstraint(
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION, name_choice=f"prod_{i}"
            ),
            default_resource_blocked={"r1": 1},
            entity=entities[i],
        )
        for i in range(len(entities))
    ]
    problem = GenericSchedulingImplProblem(
        horizon=200,
        successors=successors,
        durations_per_mode=duration_per_mode,
        non_skill_cumulative_resources={"r1": 1},
        span_blocking_constraints=span_blocking_constraints,
    )
    solver = GenericSchedulingAutoCpSatImplSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            ObjectiveHandling.SINGLE,
            objectives=[Objective.MAKESPAN],
            weights=[1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model()
    solver.cp_model.add(
        solver._starts_entity[entities[1]] == solver._starts_entity[entities[0]] + 11
    )
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        parameters_cp=ParametersCp.default(),
        time_limit=10,
    )
    sol = res[-1][0]
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


if __name__ == "__main__":
    run_constant_duration_entity_example()
