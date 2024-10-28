import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    InitialSolutionFromSolver,
    TrivialInitialSolution,
    from_solutions_to_result_storage,
)
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.decomposition import (
    MisProblem,
)
from discrete_optimization.maximum_independent_set.solvers.toulbar import (
    MisConstraintHandlerToulbar,
    ToulbarMisSolver,
    ToulbarMisSolverForLns,
)

logging.basicConfig(level=logging.INFO)


def run_toulbar_solver():
    small_example = [f for f in get_data_available() if "1dc.256.txt" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = ToulbarMisSolver(problem=mis_model)
    solver.init_model()  # (UB=-160)
    res = solver.solve(time_limit=4)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.satisfy(sol))
    print(mis_model.evaluate(sol))


def run_toulbar_solver_ws():
    small_example = [f for f in get_data_available() if "1dc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver_cpsat = CpSatMisSolver(mis_model)
    solver_cpsat.init_model()
    ws = solver_cpsat.solve(
        time_limit=5,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )[-3][0]
    solver = ToulbarMisSolver(problem=mis_model)
    solver.init_model()
    solver.set_warm_start(solution=ws)
    res = solver.solve(time_limit=300)
    sol, fit = res.get_best_solution_fit()
    print(mis_model.satisfy(sol))
    print(mis_model.evaluate(sol))


def run_toulbar_lns():
    small_example = [f for f in get_data_available() if "1dc.2048" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver_cpsat = CpSatMisSolver(mis_model)
    solver_cpsat.init_model()
    initial = TrivialInitialSolution(
        solution=from_solutions_to_result_storage(
            [mis_model.get_dummy_solution()], problem=mis_model
        )
    )
    # initial = InitialSolutionFromSolver(solver=solver_cpsat,
    #                                     time_limit=3)
    solver = ToulbarMisSolverForLns(problem=mis_model, params_objective_function=None)
    solver.init_model()
    lns = BaseLns(
        problem=mis_model,
        subsolver=solver,
        initial_solution_provider=initial,
        constraint_handler=MisConstraintHandlerToulbar(fraction_node=0.5),
    )
    res = lns.solve(
        nb_iteration_lns=10000,
        time_limit_subsolver=5,
        skip_initial_solution_provider=True,
        callbacks=[TimerStopper(total_seconds=500)],
    )
    sol = res[-1][0]
    print(mis_model.evaluate(sol))
    assert mis_model.satisfy(sol)


if __name__ == "__main__":
    run_toulbar_lns()
