#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random
from typing import Any, Optional

from discrete_optimization.generic_tasks_tools.base import TasksSolution
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorPortfolio,
    DummyConstraintExtractor,
    SchedulingConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilder,
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallbackLastRun,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.solvers.cpsat import (
    CpSatVRPTWSolver,
    Task,
    VRPTWProblem,
    VRPTWSolution,
)

TIME_LIMIT_SUBSOLVER = 5
logging.basicConfig(level=logging.INFO)


class ReinitModel(Callback):
    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        pass
        # pass
        # solver.subsolver.init_model(scaling=1, cost_per_vehicle=1000)


class NeighborBuilderVrpTw(NeighborBuilder[Task]):
    def __init__(self, problem: VRPTWProblem, fraction_vehicle: float):
        self.problem = problem
        self.fraction_vehicle = fraction_vehicle

    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        solution: VRPTWSolution = current_solution
        vehicle_used = [
            i for i in range(len(solution.routes)) if len(solution.routes[i]) > 0
        ]
        vehicle_subproblem = random.sample(
            vehicle_used, max(1, int(self.fraction_vehicle * len(vehicle_used)))
        )
        tasks_subproblem = set()
        for v in vehicle_subproblem:
            tasks_subproblem.update(set(solution.routes[v]))
        others = set(self.problem.tasks_list) - tasks_subproblem
        return tasks_subproblem, others


class VrpTaskConstraintHandler(TasksConstraintHandler):
    def extract_best_solution_from_last_iteration(
        self,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        **kwargs: Any,
    ) -> Optional[Solution]:
        if len(result_storage_last_iteration) > 0:
            sol, _ = result_storage_last_iteration[-1]
        else:
            sol, _ = result_storage[-1]
        return sol


def run_lns():
    file = [f for f in get_data_available() if "C1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    subsolver = CpSatVRPTWSolver(problem=problem)
    subsolver.init_model(scaling=1, cost_per_vehicle=1000)
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 16
    n = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderVrpTw(problem=problem, fraction_vehicle=0.4),
            NeighborBuilderSubPart(problem=problem, nb_cut_part=3),
            NeighborRandom(problem=problem, fraction_subproblem=0.5),
        ],
        weight_neighbor=[1 / 3] * 3,
        verbose=True,
    )
    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(
            plus_delta_primary=100,
            minus_delta_primary=100,
            plus_delta_secondary=10,
            minus_delta_secondary=10,
        ),
        DummyConstraintExtractor(),
    ]
    constraints_extractor = ConstraintExtractorPortfolio(
        extractors=extractors, weights=[0.8, 0.2]
    )
    constraint_handler = VrpTaskConstraintHandler(
        problem=problem,
        neighbor_builder=n,
        constraints_extractor=constraints_extractor,
    )

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
        callbacks=[
            ReinitModel(),
            WarmStartCallbackLastRun(
                warm_start_last_solution=True, warm_start_best_solution=False
            ),
        ],
        nb_iteration_lns=100,
        time_limit_subsolver_iter0=10,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
        ortools_cpsat_solver_kwargs={
            "log_search_progress": False,
            "fix_variables_to_their_hinted_value": False,
        },
    )
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_lns()
