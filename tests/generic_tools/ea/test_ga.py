#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import discrete_optimization.knapsack.parser as knapsack_parser
import discrete_optimization.tsp.parser as tsp_parser
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
    ObjectiveHandling,
)
from discrete_optimization.generic_tools.ea.nsga import Nsga


def test_binary_cx():
    files = [f for f in knapsack_parser.get_data_available() if "ks_60_0" in f]
    knapsack_problem = knapsack_parser.parse_file(files[0])

    ga_solver = Ga(
        knapsack_problem,
        crossover=DeapCrossover.CX_ONE_POINT,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        knapsack_problem,
        crossover=DeapCrossover.CX_TWO_POINT,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        knapsack_problem,
        crossover=DeapCrossover.CX_UNIFORM,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()


def test_default_ga_setting():
    files = [f for f in knapsack_parser.get_data_available() if "ks_60_0" in f]
    knapsack_problem = knapsack_parser.parse_file(files[0])
    ga_solver = Ga(
        knapsack_problem,
    )
    assert len(ga_solver._objectives) == 2
    assert ga_solver._objective_weights[1] < 0
    assert ga_solver._objectives[1] == "weight_violation"
    assert ga_solver._objective_handling == ObjectiveHandling.AGGREGATE


def test_default_nsga_setting():
    files = [f for f in knapsack_parser.get_data_available() if "ks_60_0" in f]
    knapsack_problem = knapsack_parser.parse_file(files[0])
    ga_solver = Nsga(
        knapsack_problem,
    )
    assert len(ga_solver._objectives) == 2
    assert ga_solver._objective_weights[1] < 0
    assert ga_solver._objectives[1] == "weight_violation"
    assert ga_solver._objective_handling == ObjectiveHandling.MULTI_OBJ


def test_ga_params_objective_function():
    files = [f for f in knapsack_parser.get_data_available() if "ks_60_0" in f]
    knapsack_problem = knapsack_parser.parse_file(files[0])
    ga_solver = Ga(
        knapsack_problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.SINGLE,
            objectives=["weight_violation", "value"],
            weights=[100, -1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    assert len(ga_solver._objectives) == 1
    assert ga_solver._objective_weights[0] == -100
    assert ga_solver._objective_handling == ObjectiveHandling.SINGLE

    ga_solver.solve()


def test_permutation_cx():
    files = tsp_parser.get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parser.parse_file(files[0])

    ga_solver = Ga(
        tsp_model,
        crossover=DeapCrossover.CX_ORDERED,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        crossover=DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        crossover=DeapCrossover.CX_PARTIALY_MATCHED,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()


def test_selections():
    files = tsp_parser.get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parser.parse_file(files[0])

    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_RANDOM,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_BEST,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_TOURNAMENT,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_ROULETTE,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_WORST,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()

    # selection SEL_STOCHASTIC_UNIVERSAL_SAMPLING: ! needs positive fitness to work !
    # => maximize the length here (even though not what we usually want with tsp)
    params_objective_function = ParamsObjectiveFunction(
        objectives=["length"],
        weights=[1],
        objective_handling=ObjectiveHandling.SINGLE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_STOCHASTIC_UNIVERSAL_SAMPLING,
        max_evals=1000,
        params_objective_function=params_objective_function,
    )
    kp_sol = ga_solver.solve()


def test_fully_specified_ga_setting():
    files = tsp_parser.get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parser.parse_file(files[0])

    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.SINGLE,
        objectives=["length"],
        weights=[1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    ga_solver = Ga(
        problem=tsp_model,
        encoding="permutation_from0",
        params_objective_function=params_objective_function,
        pop_size=100,
        max_evals=1000,
        crossover=DeapCrossover.CX_PARTIALY_MATCHED,
        mutation=DeapMutation.MUT_SHUFFLE_INDEXES,
        selection=DeapSelection.SEL_TOURNAMENT,
        crossover_rate=0.7,
        mut_rate=0.2,
        tournament_size=0.1,
        deap_verbose=False,
    )
    kp_sol = ga_solver.solve()


if __name__ == "__main__":
    test_binary_cx()
    test_permutation_cx()
    test_selections()
    test_default_ga_setting()
    test_fully_specified_ga_setting()
