import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.dp import DpAllocationSolver


@pytest.mark.parametrize(
    "multiobjective, symmbreak_on_used, force_allocation_when_possible",
    [
        (False, True, False),
        (False, False, False),
        (False, False, True),
        (True, True, False),
        (True, False, False),
        (True, False, True),
    ],
)
def test_dp_params(multiobjective, symmbreak_on_used, force_allocation_when_possible):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(
        instances[1], multiobjective=multiobjective
    )
    kwargs = dict(
        symmbreak_on_used=symmbreak_on_used,
        force_allocation_when_possible=force_allocation_when_possible,
    )
    solver = DpAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    # check solve
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    assert len(res) == 1
    sol = res.get_best_solution()
    allocation_problem.evaluate(sol)
    # assert allocation_problem.satisfy(sol)

    # warm start
    sol1 = allocation_problem.get_dummy_solution()
    assert not (sol.allocation == sol1.allocation)
    solver.set_warm_start(sol1)
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    assert len(res) == 1
    # assert sol.allocation == sol1.allocation  # dummy solution is not an ok solution so not the one found
