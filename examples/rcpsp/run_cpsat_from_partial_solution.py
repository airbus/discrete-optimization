#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    GenericSchedulingToRcpspTransformation,
    RcpspToGenericSchedulingTransformation,
)

# rcpsp initial pb
files_available = get_data_available()
file = [f for f in files_available if "j301_1.sm" in f][0]
rcpsp_problem = parse_file(file)

# dummy solution -> partial solution for task 1 -> 4
dummy_solution = rcpsp_problem.get_dummy_solution()
fixed_tasks = [1, 2, 3, 4]

# transformation -> generic scheduling pb
transformation = RcpspToGenericSchedulingTransformation()
generic_problem = transformation.transform_problem(rcpsp_problem)
generic_dummy_solution = transformation.forward_transform_solution(
    dummy_solution, target_problem=generic_problem
)
generic_partial_solution = generic_dummy_solution.raw_sol.take_subset(fixed_tasks)

# generic subproblem for partial solution
generic_subproblem = generic_problem.create_subproblem_from_partial_solution(
    partial_solution=generic_partial_solution
)

# 1- solve generic side using cpsat-auto
generic_solver = GenericSchedulingAutoCpSatImplSolver(problem=generic_subproblem)
generic_subsolution: GenericSchedulingImplSolution = generic_solver.solve(
    time_limit=10
).get_best_solution()
generic_full_solution = GenericSchedulingImplSolution(
    raw_sol=generic_partial_solution | generic_subsolution.raw_sol,
    problem=generic_problem,
)
rcpsp_full_solution = transformation.back_transform_solution(
    generic_full_solution, source_problem=rcpsp_problem
)
rcpsp_problem.satisfy(rcpsp_full_solution)
kpis = rcpsp_problem.evaluate(rcpsp_full_solution)
print(kpis)

# 2- solve rcpsp side using any rcpsp solver taking time windows into account (created from precedence/timelags constraints)
back_transformation = GenericSchedulingToRcpspTransformation()
rcpsp_subproblem = back_transformation.transform_problem(
    source_problem=generic_subproblem
)
rcpsp_solver = CpSatAutoRcpspSolver(problem=rcpsp_subproblem)
rcpsp_subsolution: RcpspSolution = rcpsp_solver.solve(time_limit=10).get_best_solution()
# reconstructing solution is easier generic side
generic_full_solution = GenericSchedulingImplSolution(
    raw_sol=generic_partial_solution
    | back_transformation.back_transform_solution(
        rcpsp_subsolution, source_problem=generic_subproblem
    ).raw_sol,
    problem=generic_problem,
)
rcpsp_full_solution = transformation.back_transform_solution(
    generic_full_solution, source_problem=rcpsp_problem
)
rcpsp_problem.satisfy(rcpsp_full_solution)
kpis = rcpsp_problem.evaluate(rcpsp_full_solution)
print(kpis)
