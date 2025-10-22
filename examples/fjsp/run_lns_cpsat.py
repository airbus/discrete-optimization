#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.fjsp.solvers.lns_cpsat import (
    FjspConstraintHandler,
    NeighborBuilderSubPart,
    NeighFjspConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    ParamsConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import BaseLns, ConstraintHandlerMix
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logging.basicConfig(level=logging.INFO)


class ReinitModelCallback(Callback):
    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        solver: BaseLns
        solver.subsolver.init_model()


def run_lnscpsat_fjsp():
    files = get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                FjspConstraintHandler(problem=problem, fraction_segment_to_fix=0.65),
                NeighFjspConstraintHandler(
                    problem=problem,
                    neighbor_builder=NeighborBuilderSubPart(
                        problem=problem, nb_cut_part=8
                    ),
                ),
            ],
            tag_constraint_handler=["random", "cut"],
            list_proba=[0.5, 0.5],
        ),
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=1000,
        callbacks=[
            TimerStopper(total_seconds=300),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ],
        parameters_cp=p,
        time_limit_subsolver_iter0=1,
        time_limit_subsolver=2,
    )
    sol, fit = res.get_best_solution_fit()


def run_lns_generic():
    files = get_data_available()
    file = [f for f in files if "Behnke55.fjs" in f][0]
    print(file)
    problem = parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        params_constraint_extractor=ParamsConstraintExtractor(
            minus_delta_primary=100,
            plus_delta_primary=100,
            minus_delta_secondary=20,
            plus_delta_secondary=20,
            constraint_to_current_solution_makespan=True,
            margin_rel_to_current_solution_makespan=0.05,
            fix_primary_tasks_modes=False,
            fix_secondary_tasks_modes=True,
        ),
        # constraints_extractor=ConstraintExtractorList([])
    )

    lns_solver = LnsOrtoolsCpSat(
        problem=problem, subsolver=solver, constraint_handler=constraint_handler
    )
    res = lns_solver.solve(
        callbacks=[
            # ReinitModelCallback(),
            WarmStartCallback(),
            # WarmStartCallbackLns()
        ],
        skip_initial_solution_provider=True,
        nb_iteration_lns=20,
        parameters_cp=p,
        time_limit_subsolver_iter0=10,
        time_limit_subsolver=10,
    )
    sol, fit = res.get_best_solution_fit()
    print("Evaluate", problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_lns_generic()
