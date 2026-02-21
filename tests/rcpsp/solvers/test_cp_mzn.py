#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random
import sys

import numpy as np
import pytest

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_robust import (
    AggregRcpspProblem,
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRcpspProblem,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.solution import PartialSolution, RcpspSolution
from discrete_optimization.rcpsp.solvers.cp_mzn import (
    CpMultimodeRcpspSolver,
    CpNoBoolMultimodeRcpspSolver,
    CpRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.cp_mzn_multiscenario import (
    CpMultiscenarioRcpspSolver,
)
from discrete_optimization.rcpsp.utils import plot_task_gantt

if sys.platform.startswith("win"):
    pytest.skip(reason="Much too long on windows", allow_module_level=True)


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
    solver = CpRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCp.default()
    parameters_cp.optimisation_level = optimisation_level
    result_storage = solver.solve(parameters_cp=parameters_cp, time_limit=100)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RcpspSolution(
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
    solver = CpRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCp.default()
    parameters_cp.optimisation_level = optimisation_level
    result_storage = solver.solve(parameters_cp=parameters_cp, time_limit=20)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RcpspSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)


def test_cp_sm_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    solver.init_model(output_type=True)
    result_storage = solver.solve(time_limit=5, output_type=True)
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )
    assert len(result_storage.list_solution_fits) == 15
    assert pareto_store.len_pareto_front() == 1


def create_models(
    base_rcpsp_problem: RcpspProblem, range_around_mean: int = 3, nb_models=50
):
    poisson_laws = create_poisson_laws_duration(
        base_rcpsp_problem, range_around_mean=range_around_mean
    )
    uncertain = UncertainRcpspProblem(base_rcpsp_problem, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_problem(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    average = uncertain.create_rcpsp_problem(
        MethodRobustification(MethodBaseRobustification.AVERAGE, percentile=0)
    )
    many_random_instance = [
        uncertain.create_rcpsp_problem(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(nb_models)
    ]
    many_random_instance += [
        uncertain.create_rcpsp_problem(
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
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    worst, average, many_random_instance = create_models(
        rcpsp_problem, range_around_mean=5
    )
    solver_worst = CpRcpspSolver(problem=worst)
    solver_average = CpRcpspSolver(problem=average)
    solver_original = CpRcpspSolver(problem=rcpsp_problem)
    sol_original, fit_original = solver_original.solve(
        time_limit=5
    ).get_best_solution_fit()
    sol_worst, fit_worst = solver_worst.solve(time_limit=5).get_best_solution_fit()
    sol_average, fit_average = solver_average.solve(
        time_limit=5
    ).get_best_solution_fit()
    assert fit_worst < fit_average and fit_worst < fit_original
    permutation_worst = sol_worst.rcpsp_permutation
    permutation_original = sol_original.rcpsp_permutation
    permutation_average = sol_average.rcpsp_permutation
    for instance in many_random_instance:
        sol_ = RcpspSolution(problem=instance, rcpsp_permutation=permutation_original)
        fit_original = -instance.evaluate(sol_)["makespan"]
        sol_ = RcpspSolution(problem=instance, rcpsp_permutation=permutation_average)
        fit_average = -instance.evaluate(sol_)["makespan"]
        sol_ = RcpspSolution(problem=instance, rcpsp_permutation=permutation_worst)
        fit_worst = -instance.evaluate(sol_)["makespan"]

    sol_ = RcpspSolution(problem=rcpsp_problem, rcpsp_permutation=permutation_worst)
    fit_worst = -rcpsp_problem.evaluate(sol_)["makespan"]
    sol_ = RcpspSolution(problem=rcpsp_problem, rcpsp_permutation=permutation_original)
    fit_original = -rcpsp_problem.evaluate(sol_)["makespan"]
    sol_ = RcpspSolution(problem=rcpsp_problem, rcpsp_permutation=permutation_average)
    fit_average = -rcpsp_problem.evaluate(sol_)["makespan"]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_cp_multiscenario(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_problem, range_around_mean=2)
    uncertain_model: UncertainRcpspProblem = UncertainRcpspProblem(
        base_rcpsp_problem=rcpsp_problem, poisson_laws=poisson_laws
    )
    list_rcpsp_problem = [
        uncertain_model.create_rcpsp_problem(
            MethodRobustification(
                method_base=MethodBaseRobustification.SAMPLE, percentile=0
            )
        )
        for i in range(20)
    ]
    problem = AggregRcpspProblem(
        list_problem=list_rcpsp_problem,
        method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
    )
    solver = CpMultiscenarioRcpspSolver(
        problem=problem, cp_solver_name=CpSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True, relax_ordering=False, nb_incoherence_limit=2, max_time=300
    )
    params_cp = ParametersCp.default()
    params_cp.free_search = True
    result = solver.solve(
        parameters_cp=params_cp,
        time_limit=30,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
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
        for solver_name in [CpMultimodeRcpspSolver, CpNoBoolMultimodeRcpspSolver]:
            solver = solver_name(rcpsp_problem)
            solver.init_model()
            result_storage = solver.solve(time_limit=5)
            solution = result_storage.get_best_solution()
            makespans.append(rcpsp_problem.evaluate(solution)["makespan"])
        assert makespans[0] == makespans[1]


def test_cp_mm_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpMultimodeRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    result_storage = solver.solve(time_limit=5)
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )


def test_cp_sm_partial_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j601_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpRcpspSolver(rcpsp_problem)
    dummy_solution = rcpsp_problem.get_dummy_solution()
    some_constraints = {
        task: dummy_solution.rcpsp_schedule[task]["start_time"] + 5
        for task in [1, 2, 3, 4]
    }
    partial_solution = PartialSolution(task_mode=None, start_times=some_constraints)
    solver.init_model(partial_solution=partial_solution)
    result_storage = solver.solve(time_limit=5)
    solution, fit = result_storage.get_best_solution_fit()
    assert partial_solution.start_times == {
        j: solution.rcpsp_schedule[j]["start_time"] for j in some_constraints
    }
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)


if __name__ == "__main__":
    test_cp_sm_partial_solution()
