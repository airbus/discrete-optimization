from __future__ import annotations

from typing import Any, Callable

from discrete_optimization.generic_tools.do_solver import WarmstartMixin

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var

from discrete_optimization.tsp.tsp_model import SolutionTSP, length, TSPModel2D
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.tsp.solver.tsp_solver import SolverTSP


class TSPMilpSolver(SolverTSP, GurobiMilpSolver, WarmstartMixin):

    def __init__(self, problem: TSPModel2D, **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.vars_node = None

        self.nb_var = self.problem.length_permutation * (self.problem.length_permutation + 1)

        self.edge_to_var = {}

        ind_var = 0
        for i in range(0, self.problem.length_permutation):
            self.edge_to_var[("d", i)] = ind_var
            ind_var += 1
            for j in range(0, self.problem.length_permutation):
                if i != j:
                    self.edge_to_var[(i, j)] = ind_var
                    ind_var += 1
            self.edge_to_var[(i, "a")] = ind_var
            ind_var += 1

        self.var_to_edge = {}

        ind_var = 0
        for i in range(0, self.problem.length_permutation):
            self.var_to_edge[ind_var] = ("d", i)
            ind_var += 1
            for j in range(0, self.problem.length_permutation):
                if i != j:
                    self.var_to_edge[ind_var] = (i, j)
                    ind_var += 1
            self.var_to_edge[ind_var] = (i, "a")
            ind_var += 1

    def set_warm_start(self, solution: SolutionTSP) -> None:

        """Make the solver warm start from the given solution."""
        current_point = "d"
        for i in range(self.problem.length_permutation):
            if solution.permutation_from0[0] == i:
                self.vars_node[self.edge_to_var[("d", i)]].Start = 1
                current_point = i
            else:
                self.vars_node[self.edge_to_var[("d", i)]].Start = 0
            for k in range(1, self.problem.length_permutation):
                for j in range(self.problem.length_permutation):
                    if i != j:
                        if solution.permutation_from0[k] == j and solution.permutation_from0[k-1] == i:
                            self.vars_node[self.edge_to_var[(i, j)]].Start = 1
                        else:
                            self.vars_node[self.edge_to_var[(i, j)]].Start = 0
            if solution.permutation_from0[self.problem.length_permutation-1] == i:
                self.vars_node[self.edge_to_var[(i, "a")]].Start = 1
            else:
                self.vars_node[self.edge_to_var[(i, "a")]].Start = 0

    def retrieve_current_solution(self, get_var_value_for_current_solution: Callable[[Any], float],
                                  get_obj_value_for_current_solution: Callable[[], float]) -> SolutionTSP:

        start_index = self.problem.start_index
        end_index = self.problem.end_index
        permutation = [None] * self.problem.length_permutation

        for i in range(self.problem.length_permutation):
            for j in range(self.problem.length_permutation):
                if i != j:
                    print(get_var_value_for_current_solution(self.vars_node[self.edge_to_var[(i, j)]]))

        current_point = "d"
        for i in range(self.problem.length_permutation):
            position_find = False
            j = 0
            while not position_find:
                if current_point != j:
                    if get_var_value_for_current_solution(self.vars_node[self.edge_to_var[(current_point, j)]]) >= 0.5:
                        permutation[i] = j
                        current_point = j
                        position_find = True
                j += 1

        sol = SolutionTSP(
            problem=self.problem,
            start_index=start_index,
            end_index=end_index,
            permutation_from0=permutation,
        )

        return sol

    def init_model(self, **kwargs: Any) -> None:
        # Create a new model
        self.model = Model()

        # Create variables

        self.vars_node = self.model.addVars(
            self.nb_var, vtype=GRB.BINARY, name="N"
        )

        # Set objective
        obj_exp = LinExpr()

        # add value of edge for depart point
        # only one edge ongoing of depart point
        c = LinExpr()
        for j in range(0, self.problem.length_permutation):
            obj_exp.addTerms(length(self.problem.list_points[self.problem.original_indices_to_permutation_indices[j]],
                                    self.problem.list_points[self.problem.start_index]),
                             self.vars_node[self.edge_to_var[("d", j)]])
            c.addTerms(1, self.vars_node[self.edge_to_var[("d", j)]])
        self.model.addConstr(c == 1)

        # add value of edge for arrival point
        # only one edge incoming in arrival point
        c = LinExpr()
        for j in range(0, self.problem.length_permutation):
            obj_exp.addTerms(length(self.problem.list_points[self.problem.original_indices_to_permutation_indices[j]],
                                    self.problem.list_points[self.problem.start_index]),
                             self.vars_node[self.edge_to_var[(j, "a")]])
            c.addTerms(1, self.vars_node[self.edge_to_var[(j, "a")]])
        self.model.addConstr(c == 1)

        # add value of edge between two points of the permutation
        # only one edge ongoing and incoming for each point of the permutation
        for i in range(0, self.problem.length_permutation):
            self.model.addConstr(self.vars_node[self.edge_to_var[(i, "a")]] + self.vars_node[self.edge_to_var[("d", i)]] <= 1)
            c_out = LinExpr()
            c_in = LinExpr()
            for j in range(0, self.problem.length_permutation):
                if i != j:
                    self.model.addConstr(self.vars_node[self.edge_to_var[(i, j)]] + self.vars_node[self.edge_to_var[(j, i)]] <= 1)
                    obj_exp.addTerms(
                        length(self.problem.list_points[self.problem.original_indices_to_permutation_indices[j]],
                               self.problem.list_points[self.problem.original_indices_to_permutation_indices[i]]),
                        self.vars_node[self.edge_to_var[(i, j)]])
                    c_out.addTerms(1, self.vars_node[self.edge_to_var[(i, j)]])
                    c_in.addTerms(1, self.vars_node[self.edge_to_var[(j, i)]])
            c_out.addTerms(1, self.vars_node[self.edge_to_var[(i, "a")]])
            c_in.addTerms(1, self.vars_node[self.edge_to_var[("d", i)]])
            self.model.addConstr(c_out == 1)
            self.model.addConstr(c_in == 1)

        self.model.setObjective(obj_exp, GRB.MINIMIZE)
