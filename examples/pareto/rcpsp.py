import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import (
    CpSatCumulativeResourceRcpspSolver,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    files = get_data_available()
    file = [f for f in files if "j301_4.sm" in f][0]
    problem = parse_file(file)
    problem.horizon = 300
    problem.update_functions()
    solver = CpSatCumulativeResourceRcpspSolver(problem)
    solver.init_model()
    pareto = CpsatParetoSolver(
        problem=problem,
        solver=solver,
        objective_names=["used_resource", "makespan"],
        dict_function={
            "makespan": (lambda sol: sol._internal_objectives["makespan"]),
            "used_resource": (lambda sol: sol._internal_objectives["used_resource"]),
        },
        delta_abs_improvement=[1, 1],
        delta_ref_improvement=[0, 0],
    )
    front = pareto.solve(
        [
            solver._internal_objective("used_resource"),
            solver._internal_objective("makespan"),
        ],
        time_limit=100,
        subsolver_kwargs={
            "time_limit": 4,
            "parameters_cp": ParametersCp.default_cpsat(),
        },
    )
    print(f"\nFound {len(front)} Pareto solutions:")
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        print(sol._internal_objectives)
        f1s.append(pareto.dict_function["makespan"](sol))
        f2s.append(pareto.dict_function["used_resource"](sol))
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pareto_rcpsp.png")
    plt.show()


if __name__ == "__main__":
    run_cpsat()
