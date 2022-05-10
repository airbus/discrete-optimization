from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
    ObjectiveHandling,
)
from discrete_optimization.generic_tools.path_tools import abspath_from_file
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.tsp.tsp_parser import get_data_available
from discrete_optimization.tsp.tsp_parser import parse_file as tsp_parse_file


def test_binary_cx():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])

    ga_solver = Ga(knapsack_model, crossover=DeapCrossover.CX_ONE_POINT, max_evals=1000)
    kp_sol = ga_solver.solve()

    ga_solver = Ga(knapsack_model, crossover=DeapCrossover.CX_TWO_POINT, max_evals=1000)
    kp_sol = ga_solver.solve()

    ga_solver = Ga(knapsack_model, crossover=DeapCrossover.CX_UNIFORM, max_evals=1000)
    kp_sol = ga_solver.solve()


def test_permutation_cx():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parse_file(files[0])

    ga_solver = Ga(tsp_model, crossover=DeapCrossover.CX_ORDERED, max_evals=1000)
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model, crossover=DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED, max_evals=1000
    )
    kp_sol = ga_solver.solve()

    ga_solver = Ga(
        tsp_model, crossover=DeapCrossover.CX_PARTIALY_MATCHED, max_evals=1000
    )
    kp_sol = ga_solver.solve()


def test_selections():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parse_file(files[0])

    print("SEL_RANDOM")
    ga_solver = Ga(tsp_model, selection=DeapSelection.SEL_RANDOM, max_evals=1000)
    kp_sol = ga_solver.solve()

    print("SEL_BEST")
    ga_solver = Ga(tsp_model, selection=DeapSelection.SEL_BEST, max_evals=1000)
    kp_sol = ga_solver.solve()

    print("SEL_TOURNAMENT")
    ga_solver = Ga(tsp_model, selection=DeapSelection.SEL_TOURNAMENT, max_evals=1000)
    kp_sol = ga_solver.solve()

    print("SEL_ROULETTE")
    ga_solver = Ga(tsp_model, selection=DeapSelection.SEL_ROULETTE, max_evals=1000)
    kp_sol = ga_solver.solve()

    print("SEL_WORST")
    ga_solver = Ga(tsp_model, selection=DeapSelection.SEL_WORST, max_evals=1000)
    kp_sol = ga_solver.solve()

    print("SEL_STOCHASTIC_UNIVERSAL_SAMPLING")
    ga_solver = Ga(
        tsp_model,
        selection=DeapSelection.SEL_STOCHASTIC_UNIVERSAL_SAMPLING,
        max_evals=1000,
    )
    kp_sol = ga_solver.solve()


def test_default_ga_setting():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parse_file(files[0])

    ga_solver = Ga(tsp_model)
    kp_sol = ga_solver.solve()


def test_fully_specified_ga_setting():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    tsp_model = tsp_parse_file(files[0])

    ga_solver = Ga(
        problem=tsp_model,
        encoding="permutation_from0",
        objective_handling=ObjectiveHandling.SINGLE,
        objective_weights=[-1],
        objectives=["length"],
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
