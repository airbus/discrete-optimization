#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example applying local search to a user-implemented problem."""

import logging

from tutorial_new_problem import MyKnapsackProblem, MyKnapsackSolution

from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)

logging.basicConfig(level=logging.INFO)


# instantiate a knapsack problem
problem = MyKnapsackProblem(
    max_capacity=10,
    items=[
        (2, 5),  # item 0: value=2, weight=5
        (3, 1),  # item 1: value=3, weight=1
        (2, 4),  # item 2: value=2, weight=4
        (5, 9),  # item 3: value=5, weight=9
    ],
)
# dummy solution (not taking anything)
solution = MyKnapsackSolution(
    problem=problem,
    list_taken=[
        False,
    ]
    * len(problem.items),
)

# create a mixed mutation that sample one of the available mutations for solution attributes
mixed_mutation = create_mutations_portfolio_from_problem(
    problem=problem,
    solution=solution,
)
# restart and temperature handler
restart_handler = RestartHandlerLimit(3000)
temperature_handler = TemperatureSchedulingFactor(1000, restart_handler, 0.99)

# simulated annealing solver
sa = SimulatedAnnealing(
    problem=problem,
    mutator=mixed_mutation,
    restart_handler=restart_handler,
    temperature_handler=temperature_handler,
    mode_mutation=ModeMutation.MUTATE,
)

# solve
result_storage = sa.solve(
    initial_variable=solution,
    nb_iteration_max=1000,  # increase for a more realistic problem instance
)

sol, fit = result_storage.get_best_solution_fit()

items_taken_indices = [i for i, taken in enumerate(sol.list_taken) if taken]

print(f"Best fitness: {fit}")
print(f"Taking items n°: {items_taken_indices}")

assert problem.satisfy(sol)
