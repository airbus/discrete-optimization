import logging

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpmpy.auto_impl import (
    GenericSchedulingAutoCpmpyImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_impl import (
    GenericSchedulingAutoOptalImplSolver,
)
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.transformations.generic_scheduling_impl import (
    MultiskillRcpspToGenericSchedulingTransformation,
)

logging.basicConfig(level=logging.DEBUG)


def run_solve(solver_name="optal"):
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parse_file(file, max_horizon=1000)
    transfo = MultiskillRcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(source_problem=problem)
    parameters_cp = ParametersCp.default()
    kwargs = dict(
        parameters_cp=parameters_cp,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=10,
    )
    solver = None
    match solver_name:
        case "cpsat":
            solver = GenericSchedulingAutoCpSatImplSolver(generic_problem)
        case "cpmpy":
            solver = GenericSchedulingAutoCpmpyImplSolver(generic_problem)
        case "optal":
            solver = GenericSchedulingAutoOptalImplSolver(generic_problem)

    generic_solution: GenericSchedulingImplSolution = solver.solve(
        **kwargs
    ).get_best_solution()
    assert generic_problem.satisfy(generic_solution)
    solution = transfo.back_transform_solution(generic_solution, source_problem=problem)
    assert problem.satisfy(solution)


if __name__ == "__main__":
    run_solve("optal")
