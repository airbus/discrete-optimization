import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.optal import (
    OptalSalbp12Solver,
    OptalSalbpSolver,
    SalbpProblem_1_2,
    optalcp_available,
)


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal():
    files = get_data_available()
    file = [f for f in files if "instance_n=20_10.alb" in f][0]
    problem = parse_alb_file(file)
    solver = OptalSalbpSolver(problem)
    solver.init_model(use_lb=True)
    # greedy = GreedySalbpSolver(problem)
    # sol = greedy.solve()[-1][0]
    # solver.set_warm_start(sol)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=100,
        do_not_retrieve_solutions=True,
        preset="Default",
    )


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal_on_salbp2():
    files = get_data_available()
    file = [f for f in files if "instance_n=20_10.alb" in f][0]
    problem = parse_alb_file(file)
    problem = SalbpProblem_1_2.from_salbp1(problem)
    solver = OptalSalbp12Solver(
        problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["cycle_time", "nb_stations"],
            weights=[1, 0],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = solver.solve(
        parameters_cp=p,
        do_not_retrieve_solutions=True,
        time_limit=100,
    )
