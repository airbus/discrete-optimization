from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_pareto_2d,
    plot_storage_2d,
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_MODES,
    RCPSPSolution,
)


def find_modes():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_MRCPSP_MZN_MODES(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    params_cp = ParametersCP.default()
    params_cp.nr_solutions = float("inf")
    params_cp.all_solutions = True
    result_storage = solver.solve(parameters_cp=params_cp, verbose=True)
    print(result_storage)
    print(len(result_storage))
    # for s in range(len(result_storage)):
    #    sol = RCPSPSolution(problem=rcpsp_problem,
    #                        rcpsp_permutation=list(range(rcpsp_problem.n_jobs)),
    #                        rcpsp_modes=result_storage[s][1:-1])
    # print(rcpsp_problem.satisfy(sol))


if __name__ == "__main__":
    find_modes()
