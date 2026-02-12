from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.cpsat import (
    CpSatSalbpSolver,
    ModelingCpsatSalbp,
)


def run_cpsat():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_296" in f][0]
    problem = parse_alb_file(file)
    solver = CpSatSalbpSolver(problem)
    solver.init_model(modeling=ModelingCpsatSalbp.SCHEDULING)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=100,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat()
