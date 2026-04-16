import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.singlebatch.utils import (
    GenerationMode,
    SingleBatchProcessingProblem,
    generate_random_batch_problem,
)

logging.basicConfig(level=logging.INFO)


def run_ls(problem: SingleBatchProcessingProblem, solver_ls="hc"):
    from discrete_optimization.generic_tools.mutations.mutation_integer import (
        IntegerMutation,
    )

    mutation = IntegerMutation(
        problem,
        probability_flip=0.1,
        attribute="job_to_batch",
    )

    # Create restart handler (restart after 100 iterations without improvement)
    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=100,
    )
    if solver_ls == "hc":
        # Create Hill Climber solver
        solver = HillClimber(
            problem=problem,
            mutator=mutation,
            restart_handler=restart_handler,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=False,  # Store all improving solutions
        )
    if solver_ls == "sa":
        solver = SimulatedAnnealing(
            problem=problem,
            mode_mutation=ModeMutation.MUTATE,
            mutator=mutation,
            temperature_handler=TemperatureSchedulingFactor(
                1, restart_handler, 0.999999
            ),
            restart_handler=restart_handler,
            store_solution=False,
        )

    dummy = problem.get_dummy_solution()
    res = solver.solve(
        nb_iteration_max=1000000,
        initial_variable=dummy,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    return res


def main():
    problem = generate_random_batch_problem(
        nb_jobs=100,
        capacity=100,
        size_range=(5, 30),
        duration_range=(5, 100),
        mode=GenerationMode.POSITIVE_CORRELATION,
    )
    # res = run_optal(problem)
    res = run_ls(problem, "sa")
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    main()
