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
from discrete_optimization.rcpsp.solver import PileSolverRCPSP
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


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "optimisation_level",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_cp_sm(optimisation_level):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    parameters_cp.nr_solutions = 1
    parameters_cp.optimisation_level = optimisation_level
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)


@pytest.mark.parametrize(
    "optimisation_level",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_cp_rcp(optimisation_level):
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG30/Set 1/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "Pat8.rcp" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    parameters_cp.nr_solutions = 1
    parameters_cp.optimisation_level = optimisation_level
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CPSatRCPSPSolver(problem=rcpsp_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)

    # test warm start
    start_solution = (
        PileSolverRCPSP(problem=rcpsp_problem).solve().get_best_solution_fit()[0]
    )

    # first solution is not start_solution
    assert result_storage[0][0].rcpsp_schedule != start_solution.rcpsp_schedule

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].rcpsp_schedule == start_solution.rcpsp_schedule


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools_cumulativeresource_optim(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CPSatRCPSPSolverCumulativeResource(problem=rcpsp_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 50
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools_resource_optim(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CPSatRCPSPSolverResource(problem=rcpsp_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 50
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


def test_ortools_with_cb(caplog, random_seed):
    model = "j1201_1.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CPSatRCPSPSolver(problem=rcpsp_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    parameters_cp.nr_solutions = 1

    class VariablePrinterCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.nb_solution = 0

        def on_step_end(self, step: int, res: ResultStorage, solver: CPSatRCPSPSolver):
            self.nb_solution += 1
            sol: RCPSPSolution
            sol, fit = res.list_solution_fits[-1]
            logging.debug(f"Solution #{self.nb_solution}:")
            logging.debug(sol.rcpsp_schedule)
            logging.debug(sol.rcpsp_modes)

    callbacks = [VariablePrinterCallback(), TimerStopper(2)]

    with caplog.at_level(logging.DEBUG):
        result_storage = solver.solve(callbacks=callbacks, parameters_cp=parameters_cp)

    assert "Solution #1" in caplog.text
    assert (
        "stopped by user callback" in caplog.text
    )  # stopped by timer callback instead of ortools timer
    # only true if at least one solution found after 3s (timer cb limit) and before 10s (ortools timer limit)


def test_cp_sm_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
    result_storage = solver.solve(parameters_cp=parameters_cp, output_type=True)
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )
    assert len(result_storage.list_solution_fits) == 15
    assert pareto_store.len_pareto_front() == 1


def create_models(
    base_rcpsp_model: RCPSPModel, range_around_mean: int = 3, nb_models=50
):
    poisson_laws = create_poisson_laws_duration(
        base_rcpsp_model, range_around_mean=range_around_mean
    )
    uncertain = UncertainRCPSPModel(base_rcpsp_model, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_model(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    average = uncertain.create_rcpsp_model(
        MethodRobustification(MethodBaseRobustification.AVERAGE, percentile=0)
    )
    many_random_instance = [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(nb_models)
    ]
    many_random_instance += [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.PERCENTILE, percentile=j
            )
        )
        for j in range(50, 100)
    ]
    return worst, average, many_random_instance


def test_cp_sm_robust():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    worst, average, many_random_instance = create_models(
        rcpsp_model, range_around_mean=5
    )
    solver_worst = CP_RCPSP_MZN(problem=worst)
    solver_average = CP_RCPSP_MZN(problem=average)
    solver_original = CP_RCPSP_MZN(problem=rcpsp_model)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
    sol_original, fit_original = solver_original.solve(
        parameters_cp=parameters_cp
    ).get_best_solution_fit()
    sol_worst, fit_worst = solver_worst.solve(
        parameters_cp=parameters_cp
    ).get_best_solution_fit()
    sol_average, fit_average = solver_average.solve(
        parameters_cp=parameters_cp
    ).get_best_solution_fit()
    assert fit_worst < fit_average and fit_worst < fit_original
    permutation_worst = sol_worst.rcpsp_permutation
    permutation_original = sol_original.rcpsp_permutation
    permutation_average = sol_average.rcpsp_permutation
    for instance in many_random_instance:
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_original)
        fit_original = -instance.evaluate(sol_)["makespan"]
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_average)
        fit_average = -instance.evaluate(sol_)["makespan"]
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_worst)
        fit_worst = -instance.evaluate(sol_)["makespan"]
        assert fit_worst < fit_average and fit_worst < fit_original

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_worst)
    fit_worst = -rcpsp_model.evaluate(sol_)["makespan"]
    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_original)
    fit_original = -rcpsp_model.evaluate(sol_)["makespan"]
    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_average)
    fit_average = -rcpsp_model.evaluate(sol_)["makespan"]
    assert fit_worst < fit_average and fit_worst < fit_original

    ktd = kendall_tau_similarity((sol_average, sol_worst))


def test_cp_multiscenario(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_model, range_around_mean=2)
    uncertain_model: UncertainRCPSPModel = UncertainRCPSPModel(
        base_rcpsp_model=rcpsp_model, poisson_laws=poisson_laws
    )
    list_rcpsp_model = [
        uncertain_model.create_rcpsp_model(
            MethodRobustification(
                method_base=MethodBaseRobustification.SAMPLE, percentile=0
            )
        )
        for i in range(20)
    ]
    problem = AggregRCPSPModel(
        list_problem=list_rcpsp_model,
        method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
    )
    solver = CP_MULTISCENARIO(problem=problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(
        output_type=True, relax_ordering=False, nb_incoherence_limit=2, max_time=300
    )
    params_cp = ParametersCP.default()
    params_cp.time_limit = 30
    params_cp.free_search = True
    result = solver.solve(
        parameters_cp=params_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    solution_fit = result.list_solution_fits
    objectives_cp = [sol.minizinc_obj for sol, fit in solution_fit]
    real_objective = [fit for sol, fit in solution_fit]
    assert len(solution_fit) > 0


def test_cp_mm_integer_vs_bool():
    files_available = get_data_available()
    files_to_run = [f for f in files_available if f.endswith(".mm")]
    for f in files_to_run:
        rcpsp_problem = parse_file(f)
        makespans = []
        for solver_name in [CP_MRCPSP_MZN, CP_MRCPSP_MZN_NOBOOL]:
            solver = solver_name(rcpsp_problem)
            solver.init_model()
            parameters_cp = ParametersCP.default()
            parameters_cp.time_limit = 5
            result_storage = solver.solve(parameters_cp=parameters_cp)
            solution = result_storage.get_best_solution()
            makespans.append(rcpsp_problem.evaluate(solution)["makespan"])
        assert makespans[0] == makespans[1]


def test_cp_mm_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_MRCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
    result_storage = solver.solve(parameters_cp=parameters_cp)
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )


def test_cp_sm_partial_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j601_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem)
    dummy_solution = rcpsp_problem.get_dummy_solution()
    some_constraints = {
        task: dummy_solution.rcpsp_schedule[task]["start_time"] + 5
        for task in [1, 2, 3, 4]
    }
    partial_solution = PartialSolution(task_mode=None, start_times=some_constraints)
    solver.init_model(partial_solution=partial_solution)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
    result_storage = solver.solve(parameters_cp=parameters_cp)
    solution, fit = result_storage.get_best_solution_fit()
    assert partial_solution.start_times == {
        j: solution.rcpsp_schedule[j]["start_time"] for j in some_constraints
    }
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)


if __name__ == "__main__":
    test_cp_sm_partial_solution()
