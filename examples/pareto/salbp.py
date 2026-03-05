from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.cpsat import (
    CpSatSalbp12Solver,
    SalbpProblem_1_2,
)


def run_cpsat():
    files = get_data_available()
    file = [f for f in files if "instance_n=100_337" in f][0]
    problem = parse_alb_file(file)
    problem = SalbpProblem_1_2.from_salbp1(problem)
    solver = CpSatSalbp12Solver(problem)
    solver.init_model()
    pareto = CpsatParetoSolver(solver, ["nb_stations", "cycle_time"])
    front = pareto.solve(
        [
            solver.variables["objs"]["nb_stations"],
            solver.variables["objs"]["cycle_time"],
        ]
    )
    print(f"\nFound {len(front)} Pareto solutions:")
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        f1s.append(fit["nb_stations"])
        f2s.append(fit["cycle_time"])
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig("pareto_add_remove.png")
    print("Plot saved to pareto_add_remove.png")
    print(solver.get_lexico_objectives_available())


if __name__ == "__main__":
    run_cpsat()
