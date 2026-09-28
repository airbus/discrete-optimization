#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.plot_utils import (
    plot_ressource_view,
    plot_task_gantt,
    plt,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.ortools_cpsat_tools import ParametersCp
from discrete_optimization.rcpsp_cal_preemptive.solvers.cpsat import (
    CpSatCalendarPreemptiveSolver,
)
from discrete_optimization.rcpsp_cal_preemptive.solvers.cpsat_auto import (
    CpSatAutoCalendarPreemptiveSolver,
)
from discrete_optimization.rcpsp_cal_preemptive.utils import (
    load_calendar_preemptive_rcpsp_problem,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    problem = load_calendar_preemptive_rcpsp_problem()
    solver = CpSatCalendarPreemptiveSolver(problem)
    solver.init_model()
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=15,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


def run_cpsat_auto():
    problem = load_calendar_preemptive_rcpsp_problem()
    solver = CpSatAutoCalendarPreemptiveSolver(problem)
    solver.init_model(avoid_interval_optional_for_cumulative_resources=True)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        parameters_cp=p,
        time_limit=15,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


def run_via_generic():
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
        GenericSchedulingAutoCpSatImplSolver,
    )
    from discrete_optimization.generic_tools.callbacks.loggers import (
        ProblemEvaluateLogger,
    )
    from discrete_optimization.generic_tools.transformation.transformation_solver import (
        TransformationSolver,
    )
    from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
        RcpspToGenericSchedulingTransformation,
    )

    problem = load_calendar_preemptive_rcpsp_problem()
    p = ParametersCp.default_cpsat()
    solver = TransformationSolver(
        transformation=RcpspToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            kwargs=dict(
                parameters_cp=p,
                time_limit=15,
                use_cpm_for_task_bounds=True,
                use_energy_constraints=True,
                ortools_cpsat_solver_kwargs={"log_search_progress": True},
            ),
        ),
        source_problem=problem,
    )
    callback = ProblemEvaluateLogger(
        step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
    )
    res = solver.solve(
        callbacks=[callback],
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


if __name__ == "__main__":
    run_via_generic()
