from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.top.problem import TeamOrienteeringProblem
from discrete_optimization.vrp.utils import compute_length_matrix


class TopSolver(SolverDO):
    problem: TeamOrienteeringProblem

    def __init__(
        self,
        problem: TeamOrienteeringProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        _, self.distance = compute_length_matrix(self.problem)
