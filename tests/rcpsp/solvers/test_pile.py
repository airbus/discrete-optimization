#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_robust import (
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRcpspProblem,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.pile import GreedyChoice, PileRcpspSolver
from discrete_optimization.rcpsp.utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)


def test_pile_sm():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    solver = PileRcpspSolver(problem=rcpsp_problem)
    for k in range(10):
        result_storage = solver.solve(greedy_choice=GreedyChoice.SAMPLE_MOST_SUCCESSORS)
        sol, fit = result_storage.get_best_solution_fit()
        assert rcpsp_problem.satisfy(sol)
    sol_2 = RcpspSolution(
        problem=rcpsp_problem, rcpsp_permutation=sol.rcpsp_permutation
    )
    assert rcpsp_problem.satisfy(sol_2)
    plot_ressource_view(rcpsp_problem, sol)
    plot_ressource_view(rcpsp_problem, sol_2)
    plot_resource_individual_gantt(rcpsp_problem, sol)
    assert rcpsp_problem.satisfy(sol)


def test_pile_multimode():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    solver = PileRcpspSolver(problem=rcpsp_problem)
    for k in range(10):
        result_storage = solver.solve(greedy_choice=GreedyChoice.SAMPLE_MOST_SUCCESSORS)
        sol, fit = result_storage.get_best_solution_fit()
        assert rcpsp_problem.satisfy(sol)
    sol_2 = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_modes=sol.rcpsp_modes,
        rcpsp_permutation=sol.rcpsp_permutation,
    )
    assert rcpsp_problem.satisfy(sol_2)
    plot_ressource_view(rcpsp_problem, sol)
    plot_ressource_view(rcpsp_problem, sol_2)
    plot_resource_individual_gantt(rcpsp_problem, sol)
    assert rcpsp_problem.satisfy(sol)


def test_pile_robust():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_problem)
    uncertain = UncertainRcpspProblem(rcpsp_problem, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_problem(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    solver = PileRcpspSolver(problem=worst)
    solver_original = PileRcpspSolver(problem=rcpsp_problem)
    sol_origin, fit_origin = solver_original.solve(
        greedy_choice=GreedyChoice.MOST_SUCCESSORS
    ).get_best_solution_fit()
    sol, fit = solver.solve(
        greedy_choice=GreedyChoice.MOST_SUCCESSORS
    ).get_best_solution_fit()
    assert fit <= fit_origin
    many_random_instance = [
        uncertain.create_rcpsp_problem(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(1000)
    ]
    many_random_instance = []
    many_random_instance += [
        uncertain.create_rcpsp_problem(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.PERCENTILE, percentile=j
            )
        )
        for j in range(80, 100)
    ]
    permutation = sol.rcpsp_permutation
    permutation_original = sol_origin.rcpsp_permutation

    for instance in many_random_instance:
        sol_ = RcpspSolution(problem=instance, rcpsp_permutation=permutation)
        fit = instance.evaluate(sol_)
        sol_ = RcpspSolution(problem=instance, rcpsp_permutation=permutation_original)
        fit_origin = instance.evaluate(sol_)
        assert fit_origin["makespan"] <= fit["makespan"]

    sol_ = RcpspSolution(problem=rcpsp_problem, rcpsp_permutation=permutation)
    fit = rcpsp_problem.evaluate(sol_)
    sol_ = RcpspSolution(problem=rcpsp_problem, rcpsp_permutation=permutation_original)
    fit_origin = rcpsp_problem.evaluate(sol_)
    assert fit_origin["makespan"] <= fit["makespan"]


if __name__ == "__main__":
    test_pile_multimode()
