from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver


def cpsat_single_mode_makespan_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatAutoRcpspSolver(rcpsp_problem)
    solver.init_model()
    params_cp = ParametersCp.default_cpsat()
    result = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol, fit = result.get_best_solution_fit()
    print(rcpsp_problem.evaluate(sol), rcpsp_problem.satisfy(sol))


def cpsat_multi_mode_makespan_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatAutoRcpspSolver(rcpsp_problem)
    solver.init_model()
    params_cp = ParametersCp.default_cpsat()
    result = solver.solve(
        parameters_cp=params_cp,
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol, fit = result.get_best_solution_fit()
    print(rcpsp_problem.evaluate(sol), rcpsp_problem.satisfy(sol))


if __name__ == "__main__":
    cpsat_single_mode_makespan_optimization()
