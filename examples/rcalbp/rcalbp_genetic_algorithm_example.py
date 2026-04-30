"""
RC-ALBP with Genetic Algorithm

Demonstrates how the vectorized encoding enables genetic algorithms for RC-ALBP.

The algorithm:
1. Creates a population of random task-to-station assignments
2. Evaluates each solution (cycle time + constraint penalties)
3. Selects parents based on fitness
4. Creates offspring via crossover (mix parent assignments)
5. Mutates offspring (randomly change some station assignments)
6. Repeats for multiple generations
"""

from discrete_optimization.alb.base.problem import ResourceTaskData
from discrete_optimization.alb.rcalbp.problem import RCALBPProblem, RCALBPSolution
from discrete_optimization.alb.rcalbp.utils import (
    load_rcpsp_as_albp,
    visualize_interactive_flow,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.ea.base import DeapCrossover, DeapMutation
from discrete_optimization.generic_tools.ea.ga import DeapSelection, Ga


def create_rcalbp_problem():
    """Create a small RC-ALBP instance with resource constraints."""
    # 12 tasks with varying resource requirements
    tasks_data = [
        ResourceTaskData(
            task_id=f"T{i}",
            processing_time=2 + (i % 4),
            resource_consumption={"Worker": 1 if i % 3 != 0 else 2},
        )
        for i in range(1, 13)
    ]

    # Precedence constraints (diamond structure)
    precedences = [
        ("T1", "T3"),
        ("T1", "T4"),  # T1 splits to T3 and T4
        ("T2", "T3"),
        ("T2", "T4"),  # T2 also feeds T3 and T4
        ("T3", "T7"),
        ("T4", "T8"),  # T3 and T4 go to different branches
        ("T5", "T9"),
        ("T6", "T10"),
        ("T7", "T11"),
        ("T8", "T11"),  # Merge at T11
        ("T9", "T12"),
        ("T10", "T12"),  # Merge at T12
    ]

    # 5 workstations with Worker resources
    stations = [f"WS{i}" for i in range(1, 6)]
    resources = ["Worker"]
    station_resources = {s: {"Worker": 3} for s in stations}

    # Add a shared AGV resource
    shared_resources = {"AGV"}
    shared_resource_capacities = {"AGV": 2}  # Only 2 AGVs globally

    # Some tasks need AGV
    tasks_data[0].resource_consumption["AGV"] = 1
    tasks_data[4].resource_consumption["AGV"] = 1
    tasks_data[8].resource_consumption["AGV"] = 1

    problem = RCALBPProblem(
        tasks_data=tasks_data,
        precedences=precedences,
        stations=stations,
        resources=resources,
        station_resources=station_resources,
        shared_resources=shared_resources,
        shared_resource_capacities=shared_resource_capacities,
    )

    return problem


def run_genetic_algorithm_basic():
    """Basic genetic algorithm example."""
    print("=" * 80)
    print("RC-ALBP WITH GENETIC ALGORITHM")
    print("=" * 80)
    print()
    problem = load_rcpsp_as_albp("j601_1.sm")

    print("Problem:")
    print(f"  Tasks: {problem.nb_tasks}")
    print(f"  Stations: {problem.nb_stations}")
    print(f"  Station resources: {problem.resources}")
    print(f"  Shared resources: {problem.shared_resources}")
    print(f"  Precedences: {len(problem.precedences)}")
    print()

    # Verify encoding is available
    encoding = problem.get_attribute_register()
    print("Encoding:")
    print(f"  {encoding}")
    print()

    # Configure genetic algorithm
    ga_solver = Ga(
        problem=problem,
        encoding="allocation_to_station",
        mutation=DeapMutation.MUT_UNIFORM_INT,  # Uniform integer mutation
        crossover=DeapCrossover.CX_UNIFORM,  # Uniform crossover
        selection=DeapSelection.SEL_TOURNAMENT,
        pop_size=10,  # Population size
        max_evals=1000000,  # Maximum evaluations
        mut_rate=0.2,  # Mutation rate (20%)
        crossover_rate=0.7,  # Crossover rate (70%)
        tournament_size=0.2,  # Tournament size (20% of population)
        deap_verbose=True,
    )
    ga_solver.set_warm_start(problem.get_dummy_solution())
    print("Genetic Algorithm Configuration:")
    print(f"  Population size: 50")
    print(f"  Max evaluations: 2000")
    print(f"  Mutation rate: 20%")
    print(f"  Crossover rate: 70%")
    print(f"  Selection: Tournament (size=20%)")
    print()

    # Solve with time limit
    print("Running genetic algorithm...")
    result_store = ga_solver.solve(
        callbacks=[TimerStopper(total_seconds=30)]  # 30 second time limit
    )

    # Get best solution
    best_solution: RCALBPSolution = result_store.get_best_solution()
    eval_best = problem.evaluate(best_solution)

    print()
    print("Results:")
    print(f"  Solutions evaluated: {len(result_store)}")
    print(f"  Best solution: {best_solution}")
    print(f"  Evaluation: {eval_best}")
    print(f"  Feasible: {problem.satisfy(best_solution)}")
    print()

    # Show the task assignments
    print("Best Task Assignment:")
    stations_used = {}
    for i, station_idx in enumerate(best_solution.allocation_to_station):
        task = problem.tasks[i]
        station = problem.stations[station_idx]
        if station not in stations_used:
            stations_used[station] = []
        stations_used[station].append(task)

    for station in sorted(stations_used.keys()):
        tasks = stations_used[station]
        print(f"  {station}: {tasks}")
        for task in tasks:
            start = best_solution.get_start_time_in_cycle(task)
            end = best_solution.get_end_time_in_cycle(task)
            resources = problem.get_task_data(task).resource_consumption
            print(f"    {task}: [{start}, {end}) - resources: {resources}")
    visualize_interactive_flow(problem, best_solution)

    print()


def run_genetic_algorithm_comparison():
    """Compare GA with random search."""
    print("=" * 80)
    print("GA vs RANDOM SEARCH COMPARISON")
    print("=" * 80)
    print()

    problem = create_rcalbp_problem()

    # 1. Random search baseline
    print("1. Random Search (baseline):")
    import random

    random.seed(42)

    best_random_cycle_time = float("inf")
    best_random_solution = None
    feasible_found = 0

    for _ in range(100):  # 100 random solutions
        allocation = [
            random.randint(0, problem.nb_stations - 1) for _ in range(problem.nb_tasks)
        ]
        solution = RCALBPSolution(problem=problem, allocation_to_station=allocation)
        eval_dict = problem.evaluate(solution)

        if problem.satisfy(solution):
            feasible_found += 1
            if eval_dict["cycle_time"] < best_random_cycle_time:
                best_random_cycle_time = eval_dict["cycle_time"]
                best_random_solution = solution

    print(f"   Best after 100 random trials:")
    print(f"   - Feasible solutions found: {feasible_found}/100")
    if best_random_solution:
        print(f"   - Best cycle time: {best_random_cycle_time}")
    else:
        print(f"   - No feasible solution found!")
    print()

    # 2. Genetic algorithm
    print("2. Genetic Algorithm:")
    ga_solver = Ga(
        problem=problem,
        encoding="allocation_to_station",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        crossover=DeapCrossover.CX_TWO_POINT,
        selection=DeapSelection.SEL_TOURNAMENT,
        pop_size=20,
        max_evals=100,  # Same budget as random search
        mut_rate=0.25,
        crossover_rate=0.7,
        tournament_size=0.3,
        deap_verbose=False,
    )

    result_store = ga_solver.solve()
    best_ga_solution = result_store.get_best_solution()
    eval_ga = problem.evaluate(best_ga_solution)

    print(f"   Best after 100 evaluations:")
    print(f"   - Cycle time: {eval_ga['cycle_time']}")
    print(f"   - Feasible: {problem.satisfy(best_ga_solution)}")
    print()

    # 3. Comparison
    print("3. Comparison:")
    if best_random_solution and problem.satisfy(best_ga_solution):
        improvement = (
            (best_random_cycle_time - eval_ga["cycle_time"]) / best_random_cycle_time
        ) * 100
        print(f"   GA vs Random: {improvement:+.1f}% improvement in cycle time")
        print(
            f"   GA is {'BETTER' if eval_ga['cycle_time'] < best_random_cycle_time else 'WORSE'}"
        )
    elif problem.satisfy(best_ga_solution):
        print(f"   GA found feasible solution, random search did not!")
    else:
        print(f"   Neither found a feasible solution in the budget")
    print()


def run_genetic_algorithm_evolution():
    """Track evolution over generations."""
    print("=" * 80)
    print("GA EVOLUTION TRACKING")
    print("=" * 80)
    print()

    problem = create_rcalbp_problem()

    # Small population, many generations
    ga_solver = Ga(
        problem=problem,
        encoding="allocation_to_station",
        mutation=DeapMutation.MUT_UNIFORM_INT,
        crossover=DeapCrossover.CX_UNIFORM,
        selection=DeapSelection.SEL_TOURNAMENT,
        pop_size=30,
        max_evals=600,  # 600 / 30 = 20 generations
        mut_rate=0.2,
        crossover_rate=0.8,
        tournament_size=0.25,
        deap_verbose=True,  # Show generation statistics
    )

    print("Watching evolution over 20 generations...")
    print("(Population size: 30, total evaluations: 600)")
    print()

    result_store = ga_solver.solve()

    best_solution = result_store.get_best_solution()
    eval_best = problem.evaluate(best_solution)

    print()
    print("Final Results:")
    print(f"  Cycle time: {eval_best['cycle_time']}")
    print(
        f"  Penalties: precedence={eval_best['penalty_precedence']}, "
        f"station_resources={eval_best['penalty_resource_station']}, "
        f"shared_resources={eval_best['penalty_resource_shared']}"
    )
    print(f"  Feasible: {problem.satisfy(best_solution)}")
    print()


if __name__ == "__main__":
    run_genetic_algorithm_basic()
    print("\n")
    # run_genetic_algorithm_comparison()
    print("\n")
    # run_genetic_algorithm_evolution()
