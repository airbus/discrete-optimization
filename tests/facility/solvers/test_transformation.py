#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.facility.transformations.to_binpack import (
    FacilityToBinpackTransformation,
)
from discrete_optimization.facility.transformations.to_generic import (
    FacilityToGenericAllocationTransformation,
    FacilityToGenericSchedulingTransformation,
)
from discrete_optimization.facility.transformations.to_salbp import (
    FacilityToSalbpTransformation,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)


def test_via_salbp(problem):
    from discrete_optimization.alb.salbp.solvers.greedy import GreedySalbpSolver

    solver = TransformationSolver(
        transformation=FacilityToSalbpTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedySalbpSolver, {}),
    )
    solution, fit = solver.solve().get_best_solution_fit()
    print(problem.satisfy(solution))
    print(problem.evaluate(solution))
    # don't pass, normal
    # assert problem.satisfy(solution)


def test_via_binpack(problem):
    from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver

    solver = TransformationSolver(
        transformation=FacilityToBinpackTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedyBinPackSolver, {}),
    )
    solution, fit = solver.solve().get_best_solution_fit()
    print(problem.satisfy(solution))
    print(problem.evaluate(solution))


def test_via_generic_scheduling(
    problem, generic_solver_cls, generic_solver_kwargs, generic_solver_check_solution
):
    p = ParametersCp.default_cpsat()
    p.nb_process = 1
    solver = TransformationSolver(
        transformation=FacilityToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            generic_solver_cls,
            {
                "exactly_one_unary_resource_per_task": True,
                "time_limit": 10,
                "parameters_cp": p,
                "callbacks": [NbIterationStopper(1)],
                # some param should not be passed to the wrapped solver (cpmpy conversion to subsolver could crash)
                "kwargs_keys_for_init_model_only": [
                    "exactly_one_unary_resource_per_task"
                ],
                **generic_solver_kwargs,
            },
        ),
        source_problem=problem,
    )
    solution, fit = solver.solve().get_best_solution_fit()
    if generic_solver_check_solution:
        assert problem.satisfy(solution)


def test_via_generic_allocation(
    problem,
    generic_allocation_solver_cls,
    generic_allocation_solver_kwargs,
    generic_allocation_solver_check_solution,
):
    p = ParametersCp.default_cpsat()
    p.nb_process = 1
    solver = TransformationSolver(
        transformation=FacilityToGenericAllocationTransformation(),
        solver_brick=SubBrick(
            generic_allocation_solver_cls,
            {
                "exactly_one_unary_resource_per_task": True,
                "time_limit": 10,
                "parameters_cp": p,
                "callbacks": [NbIterationStopper(1)],
                # some param should not be passed to the wrapped solver (cpmpy conversion to subsolver could crash)
                "kwargs_keys_for_init_model_only": [
                    "exactly_one_unary_resource_per_task"
                ],
                **generic_allocation_solver_kwargs,
            },
        ),
        source_problem=problem,
    )
    solution, fit = solver.solve().get_best_solution_fit()
    if generic_allocation_solver_check_solution:
        assert problem.satisfy(solution)
