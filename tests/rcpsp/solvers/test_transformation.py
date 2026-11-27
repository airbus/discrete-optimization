#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.objectives.makespan import (
    MakespanObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.resource_levels import (
    CalendarRenewableResourceLevelObjectiveComputer,
    NonRenewableResourceLevelObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat import CpSatCumulativeResourceRcpspSolver
from discrete_optimization.rcpsp.solvers.preemptive.cpsat import (
    CpSatPreemptiveRcpspSolver,
)
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    RcpspToGenericSchedulingTransformation,
)
from discrete_optimization.rcpsp.transformations.to_multiskill import (
    RcpspToMultiskillTransformation,
)
from discrete_optimization.rcpsp.transformations.to_preemptive import (
    RcpspToPreemptiveTransformation,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_via_multiskill(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = TransformationSolver(
        transformation=RcpspToMultiskillTransformation(),
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(CpSatMultiskillRcpspSolver, {}),
    )
    result_storage = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=100)
    solution: RcpspSolution
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
    assert solution.check_all_calendar_resource_capacity_constraints()


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_via_preemptive(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = TransformationSolver(
        transformation=RcpspToPreemptiveTransformation(),
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(CpSatPreemptiveRcpspSolver, {}),
    )
    result_storage = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=100)
    solution: RcpspSolution
    solution, fit = result_storage.get_best_solution_fit()
    # assert rcpsp_problem.satisfy(solution)
    # assert solution.check_all_renewable_resource_capacity_constraints()


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_via_generic_n_resources_levels_obj(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = TransformationSolver(
        transformation=RcpspToGenericSchedulingTransformation(
            list_objective_computer=[
                CalendarRenewableResourceLevelObjectiveComputer(
                    weight_objective=10000,
                ),
                NonRenewableResourceLevelObjectiveComputer(weight_objective=10000),
                MakespanObjectiveComputer(weight_objective=1),
            ]
        ),
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(GenericSchedulingAutoCpSatImplSolver, {}),
    )
    result_storage = solver.solve(time_limit=10)
    assert solver.status_solver == StatusSolver.OPTIMAL
    solution: RcpspSolution
    solution, fit = result_storage[-1]
    assert rcpsp_problem.satisfy(solution)

    def objective(sol: RcpspSolution) -> int:
        return sol.get_max_end_time() + 10000 * (
            sol.compute_aggregated_non_renewable_resources_consumptions()
            + sol.compute_aggregated_calendar_resources_levels(
                weights={
                    resource: 1
                    for resource in rcpsp_problem.non_skill_cumulative_resources_list
                }
            )
        )

    # Compare with specific rcpsp cpsat solver with cumulative resource objective
    specific_solver = CpSatCumulativeResourceRcpspSolver(problem=rcpsp_problem)
    specific_sol: RcpspSolution
    specific_sol, specific_fit = specific_solver.solve(time_limit=10)[-1]
    assert specific_solver.status_solver == StatusSolver.OPTIMAL

    print(specific_sol.rcpsp_modes, specific_sol.rcpsp_schedule)
    print(solution.rcpsp_modes, solution.rcpsp_schedule)

    assert objective(specific_sol) == objective(solution)
