import logging

logging.basicConfig(level=logging.INFO)
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.singlebatch.problem import SingleBatchProcessingProblem
from discrete_optimization.singlebatch.solvers.cpsat_auto import (
    SingleBatchAutoCpSatSolver,
)
from discrete_optimization.singlebatch.utils import (
    GenerationMode,
    generate_random_batch_problem,
)


def run_cpsat(problem: SingleBatchProcessingProblem):
    solver = SingleBatchAutoCpSatSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=50,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
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
    res = run_cpsat(problem)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    main()
