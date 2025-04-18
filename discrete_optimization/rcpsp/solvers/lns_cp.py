#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_rcpsp_tools.solvers.ls import (
    LsGenericRcpspSolver,
    LsSolverType,
)
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import PartialSolution, RcpspSolution


class PostProcessLeftShift(PostProcessSolution):
    def __init__(
        self, rcpsp_problem: RcpspProblem, partial_solution: PartialSolution = None
    ):
        self.rcpsp_problem = rcpsp_problem
        self.partial_solution = partial_solution
        if self.partial_solution is None:

            def check_solution(problem, solution):
                return True

        else:

            def check_solution(problem, solution):
                start_together = partial_solution.start_together
                start_at_end = partial_solution.start_at_end
                start_at_end_plus_offset = partial_solution.start_at_end_plus_offset
                start_after_nunit = partial_solution.start_after_nunit
                for (t1, t2) in start_together:
                    b = (
                        solution.rcpsp_schedule[t1]["start_time"]
                        == solution.rcpsp_schedule[t2]["start_time"]
                    )
                    if not b:
                        return False
                for (t1, t2) in start_at_end:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        == solution.rcpsp_schedule[t1]["end_time"]
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_at_end_plus_offset:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["end_time"] + off
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_after_nunit:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["start_time"] + off
                    )
                    if not b:
                        return False
                return True

        self.check_sol = check_solution

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        for sol in list(result_storage):
            if "satisfy" not in sol[0].__dict__.keys():
                s: RcpspSolution = sol[0]
                sol[0].satisfy = self.check_sol(
                    problem=self.rcpsp_problem, solution=s
                ) and self.rcpsp_problem.satisfy(s)
            if self.partial_solution is None:
                s: RcpspSolution = sol[0]
                solution = RcpspSolution(
                    problem=self.rcpsp_problem,
                    rcpsp_permutation=s.rcpsp_permutation,
                    rcpsp_modes=s.rcpsp_modes,
                )
                solution.satisfy = self.check_sol(
                    problem=self.rcpsp_problem, solution=solution
                ) and self.rcpsp_problem.satisfy(solution)
                result_storage.append(
                    (solution, -self.rcpsp_problem.evaluate(solution)["makespan"])
                )
        if self.partial_solution is None:
            solver = LsGenericRcpspSolver(
                problem=self.rcpsp_problem, ls_solver=LsSolverType.SA
            )
            satisfiable = [(s, f) for s, f in result_storage if s.satisfy]
            if len(satisfiable) > 0:
                s: RcpspSolution = max(satisfiable, key=lambda x: x[1])[0].copy()
            else:
                s = result_storage.get_best_solution().copy()
            s.change_problem(self.rcpsp_problem)
            result_store = solver.solve(nb_iteration_max=50, init_solution=s)
            for solution, f in result_store:
                solution.satisfy = self.check_sol(self.rcpsp_problem, solution)
                result_storage.append(
                    (solution, -self.rcpsp_problem.evaluate(solution)["makespan"])
                )
        return result_storage
