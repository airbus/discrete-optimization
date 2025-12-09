#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_permutation import (
    SwapMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.singlemachine.parser import get_data_available, parse_file

logging.basicConfig(level=logging.DEBUG)


def run_ga():
    problems = parse_file(get_data_available()[0])
    print(len(problems), " problems in the file")
    problem = problems[3]
    mixed_mutation = create_mutations_portfolio_from_problem(problem=problem)
    ga_solver = Ga(
        problem,
        encoding="permutation",
        mutation=mixed_mutation,
        pop_size=10,
        max_evals=1000000,
        mut_rate=0.1,
        crossover_rate=0.9,
        tournament_size=0.2,  #
    )
    results = ga_solver.solve()
    sol, fit = results.get_best_solution_fit()
    print(problem.evaluate(sol))


def run_ls():
    from discrete_optimization.generic_tools.ls.simulated_annealing import (
        SimulatedAnnealing,
    )

    problems = parse_file(get_data_available()[0])
    print(len(problems), " problems in the file")
    problem = problems[2]
    solution = problem.get_dummy_solution()
    res = RestartHandlerLimit(3000)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=problem, selected_mutations={SwapMutation}
    )

    sa = SimulatedAnnealing(
        problem=problem,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=10000, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE,
    )
    res = sa.solve(nb_iteration_max=10000000, initial_variable=solution)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_ga()
