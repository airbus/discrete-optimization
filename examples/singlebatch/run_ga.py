import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapSelection,
    Ga,
)
from discrete_optimization.singlebatch.utils import (
    GenerationMode,
    SingleBatchProcessingProblem,
    generate_random_batch_problem,
)

logging.basicConfig(level=logging.INFO)


def run_ga(problem: SingleBatchProcessingProblem):
    solver = Ga(
        problem,
        crossover=DeapCrossover.CX_UNIFORM,
        selection=DeapSelection.SEL_TOURNAMENT,
        encoding="job_to_batch",
        pop_size=30,
        max_evals=1000000,
        mut_rate=0.1,
        crossover_rate=0.1,
        tournament_size=2,
        deap_verbose=True,
    )
    dummy = problem.get_dummy_solution()
    solver.set_warm_start(dummy)
    res = solver.solve(
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ]
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
    res = run_ga(problem)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    main()
