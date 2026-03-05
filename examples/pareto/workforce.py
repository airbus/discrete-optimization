from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model()

    pareto = CpsatParetoSolver(solver, ["nb_teams", "duration"])
    front = pareto.solve(
        [solver.variables["objs"]["nb_teams"], solver.variables["objs"]["duration"]]
    )
    print(f"\nFound {len(front)} Pareto solutions:")
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        f1s.append(fit["nb_teams"])
        f2s.append(fit["duration"])
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
