import os

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LNS_CP
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.cp_solvers import (
    CPKnapsackMZN2,
    KnapsackModel,
)
from discrete_optimization.knapsack.solvers.knapsack_lns_cp_solver import (
    ConstraintHandlerKnapsack,
    InitialKnapsackMethod,
    InitialKnapsackSolution,
)


def test_knapsack_lns():
    model_file = [f for f in get_data_available() if "ks_30_0" in f][
        0
    ]  # optim result "54939"
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_cp = ParametersCP.default()
    params_cp.TimeLimit = 10
    params_cp.TimeLimit_iter0 = 1
    solver = CPKnapsackMZN2(
        model,
        cp_solver_name=CPSolverName.CHUFFED,
        params_objective_function=params_objective_function,
    )
    solver.init_model()
    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.83)
    lns_solver = LNS_CP(
        problem=model,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store_pure_cp = solver.solve(parameters_cp=params_cp)
    solution_pure_cp = result_store_pure_cp.get_best_solution_fit()
    print("Pure CP : ", solution_pure_cp[0], solution_pure_cp[1])
    result_store = lns_solver.solve_lns(
        parameters_cp=params_cp, nb_iteration_lns=200, max_time_seconds=30
    )
    solution = result_store.get_best_solution_fit()[0]
    print([x[1] for x in result_store.list_solution_fits])
    print(solution)
    print("Satisfy : ", model.satisfy(solution))
    print(model.evaluate(solution))
    import matplotlib.pyplot as plt

    fitness = [f for s, f in result_store.list_solution_fits]
    fig, ax = plt.subplots(1)
    ax.plot(fitness)
    ax.set_title("Fitness evolution ")


if __name__ == "__main__":
    test_knapsack_lns()
