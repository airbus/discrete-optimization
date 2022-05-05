import os

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.knapsack.knapsack_parser import files_available, parse_file
from discrete_optimization.knapsack.solvers.knapsack_lns_solver import (
    ConstraintHandlerKnapsack,
    InitialKnapsackMethod,
    InitialKnapsackSolution,
)
from discrete_optimization.knapsack.solvers.lp_solvers import (
    KnapsackModel,
    KnapsackSolution,
    LPKnapsack,
    MilpSolverName,
    ParametersMilp,
)


def knapsack_lns():
    model_file = [f for f in files_available if "ks_500_0" in f][0]
    model: KnapsackModel = parse_file(model_file)
    params_objective_function = get_default_objective_setup(problem=model)
    params_milp = ParametersMilp(
        time_limit=300,
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
        n_solutions_max=1000,
    )
    solver = LPKnapsack(
        model,
        milp_solver_name=MilpSolverName.GRB,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    result_lp = solver.solve(parameters_milp=params_milp)
    print(result_lp.get_best_solution_fit())

    initial_solution_provider = InitialKnapsackSolution(
        problem=model,
        initial_method=InitialKnapsackMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerKnapsack(problem=model, fraction_to_fix=0.99)
    lns_solver = LNS_MILP(
        problem=model,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve_lns(
        parameters_milp=params_milp, nb_iteration_lns=10000
    )
    solution = result_store.get_best_solution_fit()[0]
    print([x[1] for x in result_store.list_solution_fits])
    print(solution)
    print("Satisfy : ", model.satisfy(solution))
    print(model.evaluate(solution))


if __name__ == "__main__":
    knapsack_lns()
