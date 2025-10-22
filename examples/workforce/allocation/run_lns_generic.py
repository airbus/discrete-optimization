#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ALLOCATION_OBJECTIVES,
    SCHEDULING_OBJECTIVES,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.workforce.allocation.parser import (
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.scheduling.parser import get_data_available

logging.basicConfig(level=logging.INFO)
TIME_LIMIT_SUBSOLVER = 5


def run_lns_cpsat():
    instance = [p for p in get_data_available() if "instance_68.json" in p][0]
    problem = parse_to_allocation_problem(instance, multiobjective=True)
    subsolver = CpsatTeamAllocationSolver(
        problem=problem,
    )
    subsolver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    subsolver.set_model_obj_aggregated([("nb_teams", 1000), ("duration", 1)])
    parameters_cp = ParametersCp.default()
    extractors: list[BaseConstraintExtractor] = [
        NbChangesAllocationConstraintExtractor(),
        NbUsagesAllocationConstraintExtractor(),
        SubresourcesAllocationConstraintExtractor(),
        SubtasksAllocationConstraintExtractor(fix_secondary_tasks_allocation=True),
    ]
    constraints_extractor = ConstraintExtractorList(extractors=extractors)

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        constraints_extractor=constraints_extractor,
    )

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
        callbacks=[WarmStartCallback()],
        nb_iteration_lns=20,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        time_limit_subsolver_iter0=1,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


if __name__ == "__main__":
    run_lns_cpsat()
