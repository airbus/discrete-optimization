#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True
from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver


class ToulbarBinPackSolver(ToulbarSolver, WarmstartMixin):
    problem: BinPackProblem

    def init_model(self, **kwargs) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        number_items = self.problem.nb_items
        neighs_dict = {}
        for i in range(number_items):
            edges = [
                x for x in self.problem.incompatible_items if x[0] == i or x[1] == i
            ]
            neighs = [x[0] if x[0] != i else x[1] for x in edges]
            neighs_dict[i] = neighs

        # we don't have to have a very tight bound.
        model = pytoulbar2.CFN(number_items, vns=kwargs["vns"])
        model.AddVariable("nb_bins", range(number_items))
        model.AddFunction(["nb_bins"], range(number_items))
        for i in range(number_items):
            model.AddVariable(f"x_{i}", range(i + 1))
            model.AddLinearConstraint([1, -1], [f"x_{i}", "nb_bins"], "<=", 0)
            # encode that x_{i}<=nb_bins.
        for i in range(number_items):
            neighs = neighs_dict[i]
            if len(neighs) > 0:
                for neigh in neighs:
                    model.AddFunction(
                        [f"x_{i}", f"x_{neigh}"],
                        [
                            1000 if i == j else 0
                            for i in range(i + 1)
                            for j in range(neigh + 1)
                        ],
                    )

        for i in range(number_items):
            model.AddGeneralizedLinearConstraint(
                [
                    (f"x_{j}", i, self.problem.list_items[j].weight)
                    for j in range(i, number_items)
                ],
                "<=",
                self.problem.capacity_bin,
            )
        self.model = model
        if hasattr(self, "sol") and kwargs["greedy_start"]:
            self.set_warm_start(self.sol)

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> BinPackSolution:
        return BinPackSolution(
            problem=self.problem,
            allocation=solution_from_toulbar2[0][1 : 1 + self.problem.nb_items],
        )

    def set_warm_start(self, solution: BinPackSolution) -> None:
        max_bin = max(solution.allocation)
        self.model.CFN.wcsp.setBestValue(0, max_bin)
        for i in range(1, self.problem.nb_items + 1):
            self.model.CFN.wcsp.setBestValue(i, solution.allocation[i - 1])
