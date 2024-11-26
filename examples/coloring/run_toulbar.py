#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import plot_coloring_solution, plt
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringSolution,
    transform_coloring_problem,
)
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.toulbar import (
    ColoringConstraintHandlerToulbar,
    ToulbarColoringSolver,
    ToulbarColoringSolverForLns,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    InitialSolutionFromSolver,
    TrivialInitialSolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    from_solutions_to_result_storage,
)
from discrete_optimization.generic_tools.toulbar_tools import to_lns_toulbar

logging.basicConfig(level=logging.INFO)


def run_toulbar_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_100_9" in f][0]
    color_problem = parse_file(file)
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=50,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=1,
    )
    # solver.model.Dump("test.wcsp")
    result_store = solver.solve(time_limit=100)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    plt.show()
    assert color_problem.satisfy(solution)


def run_toulbar_coloring_with_ws():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_250_9" in f][0]
    color_problem = parse_file(file)
    solv = GreedyColoringSolver(problem=color_problem)
    greedy_sol = solv.solve(
        strategy=NxGreedyColoringMethod.best
    ).get_best_solution_fit()[0]
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=None,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=1,
        # vns=-4
    )
    solver.set_warm_start(greedy_sol)
    result_store = solver.solve(time_limit=100)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    plt.show()
    assert color_problem.satisfy(solution)


def run_toulbar_lns_manual():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_500_7" in f][0]
    color_problem = parse_file(file)
    solv = GreedyColoringSolver(problem=color_problem)
    greedy_sol = solv.solve(
        strategy=NxGreedyColoringMethod.best
    ).get_best_solution_fit()[0]
    # greedy_sol = color_problem.get_dummy_solution()
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=len(set(greedy_sol.colors)),
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=1,
    )
    # solver.model.Dump("gc_250_9.wcsp")
    solver.model.SolveFirst()
    depth = solver.model.Depth()
    current_sol: ColoringSolution = greedy_sol
    fits = []
    # solver.model.configuration = True
    solver.model.CFN.timer(100)

    for k in range(100):
        solver.model.Store()
        solver.model.CFN.timer(100)
        max_ = max(current_sol.colors)
        random_indexes = random.sample(
            range(1, color_problem.number_of_nodes + 1),
            k=int(0.25 * color_problem.number_of_nodes),
        )
        text = ",".join(
            f"{index}={current_sol.colors[index-1]}"
            for index in random_indexes
            if current_sol.colors[index - 1] < max_
        )
        text = "," + text
        solver.model.Parse(text)
        # solver.model.MultipleAssign(random_indexes, [current_sol.colors[index-1]
        #                                               for index in random_indexes])
        solver.set_warm_start(current_sol)
        try:
            soluce = solver.model.SolveNext(showSolutions=1, timeLimit=10)
            print("Dual bound :", solver.model.GetDDualBound())
            current_sol = ColoringSolution(
                problem=color_problem,
                colors=soluce[0][1 : 1 + color_problem.number_of_nodes],
            )
            # result_store = solver.solve(time_limit=10)
            # current_sol = result_store.get_best_solution_fit()[0]
            print(color_problem.satisfy(current_sol))
            print(color_problem.evaluate(current_sol))
            fits.append(solver.aggreg_from_sol(current_sol))
        except Exception as e:
            print("FAIL", e)
        solver.model.Restore(depth)

    print(fits)
    # plot_coloring_solution(solution)
    # plt.show()
    assert color_problem.satisfy(current_sol)


def run_toulbar_with_do_lns():
    logging.basicConfig(level=logging.INFO)
    from discrete_optimization.generic_tools.lns_tools import (
        BaseLns,
        InitialSolutionFromSolver,
    )

    file = [f for f in get_data_available() if "gc_250_9" in f][0]
    color_problem = parse_file(file)
    initial = InitialSolutionFromSolver(
        solver=GreedyColoringSolver(problem=color_problem),
        strategy=NxGreedyColoringMethod.best,
    )
    # solver = ToulbarColoringSolverForLns(color_problem, params_objective_function=None)
    solver = to_lns_toulbar(ToulbarColoringSolver)(color_problem)
    solver.init_model(
        nb_colors=None,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=1,
    )
    lns = BaseLns(
        problem=color_problem,
        subsolver=solver,
        initial_solution_provider=initial,
        constraint_handler=ColoringConstraintHandlerToulbar(fraction_node=0.4),
    )
    res = lns.solve(
        nb_iteration_lns=100,
        time_limit_subsolver=5,
        callbacks=[TimerStopper(total_seconds=200)],
    )
    sol = res[-1][0]
    assert color_problem.satisfy(sol)


def run_toulbar_with_constraints():
    file = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(file)
    color_problem = transform_coloring_problem(
        color_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ColoringConstraints(color_constraint={0: 3, 1: 2, 2: 4}),
    )
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=20,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=2,
    )
    result_store = solver.solve(time_limit=10)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    assert color_problem.satisfy(solution)
    plt.show()


def run_optuna_study():
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    files_available = get_data_available()
    file = [f for f in get_data_available() if "gc_250_9" in f][0]
    color_problem = parse_file(file)
    solvers_to_test = [ToulbarColoringSolver, BaseLns]
    copy = ToulbarColoringSolver.hyperparameters
    ToulbarColoringSolver.hyperparameters = (
        ToulbarColoringSolver.copy_and_update_hyperparameters(
            ["vns", "value_sequence_chain", "greedy_start"],
            **{
                "vns": {"choices": [None, -4]},
                "value_sequence_chain": {"choices": [False]},
                "greedy_start": {"choices": [True]},
            },
        )
    )
    ToulbarColoringSolver.hyperparameters += [
        c for c in copy if c.name not in ["vns", "value_sequence_chain", "greedy_start"]
    ]
    generic_optuna_experiment_monoproblem(
        problem=color_problem,
        study_basename="study-toulbar",
        # storage_path="./optuna-journal-toulbar.log",
        solvers_to_test=solvers_to_test,
        overwrite_study=True,
        kwargs_fixed_by_solver={
            ToulbarColoringSolver: {"time_limit": 50, "greedy_start": True},
            BaseLns: {
                "callbacks": [TimerStopper(total_seconds=50)],
                "constraint_handler": ColoringConstraintHandlerToulbar(
                    fraction_node=0.83
                ),
                "post_process_solution": None,
                "initial_solution_provider": InitialSolutionFromSolver(
                    solver=GreedyColoringSolver(problem=color_problem),
                    strategy=NxGreedyColoringMethod.best,
                ),
                "nb_iteration_lns": 1000,
                "time_limit_subsolver": 5,
            },
        },
        suggest_optuna_kwargs_by_name_by_solver={
            BaseLns: {"subsolver": {"choices": [to_lns_toulbar(ToulbarColoringSolver)]}}
        },
    )


if __name__ == "__main__":
    run_optuna_study()
