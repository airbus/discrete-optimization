#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ChainingConstraintExtractor,
    ConstraintExtractorList,
    ConstraintExtractorPortfolio,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
    WarmStartCallbackLastRun,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
    ObjectivesEnum,
)

TIME_LIMIT_SUBSOLVER = 5
logging.basicConfig(level=logging.INFO)


def run_lns():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    subsolver = CPSatAllocSchedulingSolver(
        problem=problem,
    )
    subsolver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION]
    )
    parameters_cp = ParametersCp.default()

    extractors: list[BaseConstraintExtractor] = [
        # SchedulingConstraintExtractor(),
        NbChangesAllocationConstraintExtractor(5),
        # This limited neighbor could help reduce the dispersion workload
        # NbUsagesAllocationConstraintExtractor(minus_delta_nb_usages_per_unary_resource=3,
        #                                       plus_delta_nb_usages_per_unary_resource=3),
        # The number of tasks allocated to teams remains a bit stable
        SubresourcesAllocationConstraintExtractor(0.5),
        # half of the resource will have the same allocated tasks
        SubtasksAllocationConstraintExtractor(
            # This should help the nb-teams optim ?
            fix_secondary_tasks_allocation=True,
            frac_random_fixed_tasks=0.8,
        ),
    ]
    constraints_extractor = ConstraintExtractorPortfolio(extractors=extractors)

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
        callbacks=[
            WarmStartCallbackLastRun(
                warm_start_last_solution=True, warm_start_best_solution=False
            )
        ],
        nb_iteration_lns=100,
        time_limit_subsolver_iter0=1,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_lns()
