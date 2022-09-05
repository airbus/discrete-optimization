from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.result_storage.result_storage import (
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    PartialSolution,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    kendall_tau_similarity,
    plot_task_gantt,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_NOBOOL,
    CP_RCPSP_MZN,
)


def test_cp_sm():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_5.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 5
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


def create_models(base_rcpsp_model: RCPSPModel, range_around_mean: int = 3):
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
        for i in range(50)
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
    solver_worst = CP_RCPSP_MZN(rcpsp_model=worst)
    solver_average = CP_RCPSP_MZN(rcpsp_model=average)
    solver_original = CP_RCPSP_MZN(rcpsp_model=rcpsp_model)
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
