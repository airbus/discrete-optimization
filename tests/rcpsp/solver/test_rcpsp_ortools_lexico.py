#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random

import numpy as np
import pytest

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
    TimerStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
)
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import PartialSolution, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    kendall_tau_similarity,
    plot_task_gantt,
)
from discrete_optimization.rcpsp.robust_rcpsp import (
    AggregRCPSPModel,
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_NOBOOL,
    CP_RCPSP_MZN,
)
from discrete_optimization.rcpsp.solver.cp_solvers_multiscenario import CP_MULTISCENARIO
from discrete_optimization.rcpsp.solver.cpsat_solver import (
    CPSatRCPSPSolver,
    CPSatRCPSPSolverCumulativeResource,
    CPSatRCPSPSolverResource,
)


@pytest.mark.parametrize(
    "objectives",
    [
        None,
        ["makespan", "used_resource"],
        ("used_resource", "makespan"),
    ],
)
def test_ortools_cumulativeresource_optim(objectives):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    subsolver = CPSatRCPSPSolverCumulativeResource(problem=rcpsp_problem)

    solver = LexicoSolver(
        problem=rcpsp_problem,
        subsolver=subsolver,
    )
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10

    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        objectives=objectives,
    )

    print([sol._intern_objectives for sol, fit in result_storage.list_solution_fits])

    # for obj in subsolver.get_model_objectives_available():
    #     obj_array = np.array(
    #         [
    #             sol._intern_objectives[obj]
    #             for sol, fit in result_storage.list_solution_fits
    #         ]
    #     )
    #     assert (obj_array[1:] <= obj_array[:-1]).all()


@pytest.mark.parametrize(
    "objectives",
    [
        None,
        ["makespan", "used_resource"],
        ("used_resource", "makespan"),
    ],
)
def test_ortools_resource_optim(objectives):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    subsolver = CPSatRCPSPSolverResource(problem=rcpsp_problem)

    solver = LexicoSolver(
        problem=rcpsp_problem,
        subsolver=subsolver,
    )
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10

    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        objectives=objectives,
    )

    print([sol._intern_objectives for sol, fit in result_storage.list_solution_fits])

    # for obj in subsolver.get_model_objectives_available():
    #     obj_array = np.array(
    #         [
    #             sol._intern_objectives[obj]
    #             for sol, fit in result_storage.list_solution_fits
    #         ]
    #     )
    #     assert (obj_array[1:] <= obj_array[:-1]).all()
