import pytest

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ALLOCATION_OBJECTIVES,
    SCHEDULING_OBJECTIVES,
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver


@pytest.mark.parametrize(
    "objective_subproblem",
    SCHEDULING_OBJECTIVES + (ObjectiveSubproblem.INITIAL_OBJECTIVE,),
)
def test_lns(objective_subproblem):
    problem = parse_file(get_data_available()[0])
    subsolver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = solver.solve(
        nb_iteration_lns=20,
        time_limit_subsolver=5,
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize("objective_subproblem", ALLOCATION_OBJECTIVES)
def test_lns_obj_nok(objective_subproblem):
    problem = parse_file(get_data_available()[0])
    subsolver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)

    with pytest.raises(ValueError):
        TasksConstraintHandler(
            problem=problem,
            objective_subproblem=objective_subproblem,
        )
