import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.singlebatch.solvers.cpsat import (
    CpSatSingleBatchSolver,
    ModelingBpm,
)
from discrete_optimization.singlebatch.solvers.dp import DpSingleBatchSolver
from discrete_optimization.singlebatch.solvers.lp import MathOptSingleBatchSolver
from discrete_optimization.singlebatch.solvers.optal import OptalSingleBatchSolver
from discrete_optimization.singlebatch.utils import (
    GenerationMode,
    SingleBatchProcessingProblem,
    generate_random_batch_problem,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat(problem: SingleBatchProcessingProblem):
    solver = CpSatSingleBatchSolver(problem)
    solver.init_model(modeling=ModelingBpm.BINARY)
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=30,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    return res


def run_optal(problem: SingleBatchProcessingProblem):
    solver = OptalSingleBatchSolver(problem)
    solver.init_model(modeling=ModelingBpm.BINARY)
    p = ParametersCp.default_cpsat()
    import optalcp as cp

    p.nb_process = 12
    lns = cp.WorkerParameters(
        _packPropagationLevel=2,
        cumulPropagationLevel=3,
        _itvMappingPropagationLevel=2,
        searchType="LNS",
    )
    fds = cp.WorkerParameters(
        _packPropagationLevel=2,
        cumulPropagationLevel=3,
        _itvMappingPropagationLevel=2,
        searchType="FDS",
    )
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=100,
        # workers=[lns]*4+[fds]*4,
        preset="Large",
    )
    return res


def run_dp(problem: SingleBatchProcessingProblem):
    solver = DpSingleBatchSolver(problem)
    solver.init_model()
    res = solver.solve(
        solver="LNBS",
        threads=4,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=30,
    )
    return res


def run_mathopt(problem: SingleBatchProcessingProblem):
    from discrete_optimization.singlebatch.solvers.lp import BpmLpFormulation

    solver = MathOptSingleBatchSolver(problem)
    solver.init_model(formulation=BpmLpFormulation.NAIVE)
    p = ParametersMilp.default()
    from discrete_optimization.generic_tools.lp_tools import mathopt

    res = solver.solve(
        parameters_milp=p,
        mathopt_solver_type=mathopt.SolverType.CP_SAT,
        time_limit=30,
        store_mathopt_res=True,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        extract_solutions_from_mathopt_res=True,
        mathopt_enable_output=True,
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
