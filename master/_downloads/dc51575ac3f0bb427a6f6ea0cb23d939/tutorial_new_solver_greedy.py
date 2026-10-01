from typing import Any, Optional

from tutorial_new_problem import Item, MyKnapsackProblem, MyKnapsackSolution

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class MyKnapsackGreedySolver(SolverDO):
    """Greedy solver class for MyKnapsackProblem.

    This solver sort the items by density (value/weight)
    and take them in this order until the max capacity is reached.

    """

    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        """Initialize the solver.

        Args:
            problem: problem to solve
            params_objective_function: define the objective to optimize
                and how to aggregate them (or not aggregate them for multiobjective solvers).
                By default, constructed from `problem.get_objective_register()`.
            **kwargs:

        """
        if not isinstance(problem, MyKnapsackProblem):
            raise ValueError(
                "`MyKnapsackGreedySolver` can only handle `MyKnapsackProblem`."
            )
        super().__init__(problem, params_objective_function, **kwargs)

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Solve the problem.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: any argument specific to the solver

        Returns (ResultStorage): a result object containing a pool of solutions
            to the problem

        """
        # first call to callbacks
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        # Sort items by density=value/weight  (discard items overcoming the max capacity)
        def compute_density(item: Item) -> float:
            value, weight = item
            return value / weight

        i_items_by_density = sorted(
            (
                i_item
                for i_item, (value, weight) in enumerate(self.problem.items)
                if weight <= self.problem.max_capacity
            ),
            key=lambda i_item: compute_density(self.problem.items[i_item]),
            reverse=True,
        )

        # take items until reaching max capacity
        total_weight = 0
        step = 0
        list_taken = [
            False,
        ] * len(self.problem.items)
        res = (
            self.create_result_storage()
        )  # empty result storage (to be consumed by callbacks)
        for i_item in i_items_by_density:
            value, weight = self.problem.items[i_item]
            total_weight += weight
            if total_weight > self.problem.max_capacity:
                break
            else:
                list_taken[i_item] = True
                # contruct partial solution (copy the list to avoid ending with same solutions)
                sol = MyKnapsackSolution(
                    problem=self.problem, list_taken=list(list_taken)
                )
                # compute aggregated fitness
                fit = self.aggreg_from_sol(sol)
                # store the (sol, fit) tuple into the result storage
                res.append((sol, fit))
                # intermediate call to callbacks
                callbacks_list.on_step_end(step=step, res=res, solver=self)
                step += 1

        # final call to callbacks
        callbacks_list.on_solve_end(res=res, solver=self)

        return res


if __name__ == "__main__":
    import logging

    from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger

    logging.basicConfig(level=logging.DEBUG)

    # instantiate a knapsack problem
    problem = MyKnapsackProblem(
        max_capacity=10,
        items=[
            (2, 5),  # item 0: value=2, weight=5
            (3, 1),  # item 1: value=3, weight=1
            (2, 4),  # item 2: value=2, weight=4
            (5, 9),  # item 3: value=5, weight=9
        ],
    )

    # instantiate the greedy solver
    solver = MyKnapsackGreedySolver(problem=problem)

    # solve with a logging callback
    result_storage = solver.solve(callbacks=[ObjectiveLogger()])

    # display best solution
    sol, fit = result_storage.get_best_solution_fit()
    items_taken_indices = [i for i, taken in enumerate(sol.list_taken) if taken]
    print(f"Best fitness: {fit}")
    print(f"Taking items n°: {items_taken_indices}")

    # check solution satisfies the problem
    assert problem.satisfy(sol)
