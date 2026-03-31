# Greedy solver

To showcase how a solver can be created directly from the base class,
we will implement a simple greedy solver.


## Creating the skeleton

We start by initiating a solver class by simply deriving from the base class `SolverDO`:

```python
from discrete_optimization.generic_tools.do_solver import SolverDO


class MyKnapsackGreedySolver(SolverDO): ...

```

Then, as before, use your favorite IDE to generate the methods to implement (or look in the source file of the base classes for abtract methods).
You should get something like:

```python
from typing import Optional, Any

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class MyKnapsackGreedySolver(SolverDO):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        pass

```

:::{tip}
- with PyCharm, right-click on the class, and select "Generate..." > "Implement Methods..."
- with VSCode, click on the class name, then on the lightbulb that comes up above, then "Implement all inherited abstract classes"
:::

## Implementation

### First version

The only mandatory method to implement is `solve()`.
In a greedy approach, the item with the higher ratio value/weight is selected and added to the solution.
The process repeats until the max capacity is reached.

Note that `SolverDO` already implements an `__init__()` method that
stores the problem in the `problem` attribute and build the methods computing and aggregating the objectives.
You could still override it to ensure dealing with a MyKnapsackProblem, but we keep it simple here.
We only specify the type of the `problem` attribute to help the IDE to correctly type it.

```python
class MyKnapsackGreedySolver(SolverDO):
    """Greedy solver class for MyKnapsackProblem.

    This solver sort the items by density (value/weight)
    and take them in this order until the max capacity is reached.

    """

    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Solve the problem

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: any argument specific to the solver

        Returns: a result object containing a pool of solutions to the problem

        """

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

        # Take items until reaching max capacity
        total_weight = 0
        list_taken = [
            False,
        ] * len(self.problem.items)
        for i_item in i_items_by_density:
            value, weight = self.problem.items[i_item]
            total_weight += weight
            if total_weight > self.problem.max_capacity:
                break
            else:
                list_taken[i_item] = True

        # Contruct solution
        sol = MyKnapsackSolution(problem=self.problem, list_taken=list_taken)

        # Compute aggregated fitness
        fit = self.aggreg_from_sol(sol)

        # Construct result_storage (with only one solution but could contain more for other solvers)
        res = self.create_result_storage(list_solution_fits=[(sol, fit)])

        return res
```

Then solving the problem with it will sum up to
```python
solver = MyKnapsackGreedySolver(problem=problem)
solution = solver.solve().get_best_solution()
```



### Adding callbacks support


Note that the method `solve()` takes a list of callbacks as argument to allow a user to hook
at different points of the solving process.
See [this notebook](../notebooks.md#callbacks-usage) for more information about callbacks.

To allow the callbacks mechanics we should:
- wrap the callbacks into a `CallbackList` to call the whole list at once
- call `on_solve_start()` at solve start
- call `on_solve_end()` at solve end
- call `on_step_end()` at end of each step for an iterative solver,
  usually after each new solution found

To see it in action, we will store a partial solution each time a new item is added:
```python
class MyKnapsackGreedySolver(SolverDO):

    ...

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
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
```

We  can solve the problem with a callback logging the objective at each step via:
```python
solver = MyKnapsackGreedySolver(problem=problem)
solution = solver.solve(callbacks=[ObjectiveLogger()]).get_best_solution()
```


## Solver in action

The resulting script can be found here: <path:tutorial_new_solver_greedy.py>, with an example of how to use it with a callback
logging the objective at each iteration. It should be run in the same directory as the previous module
<path:tutorial_new_problem.py> that declares the knapsack problem and solution classes, so that they can be imported.
