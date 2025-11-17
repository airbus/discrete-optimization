# How to write its own problem class?

Let us first try to implement the problem class (and associated solution class). If your need is to implement
a solver for an already existing problem, go directly to the [next section](#how-to-write-its-own-solver-class).


## Creating the skeletons

We start by initiating a problem and solution class that will be used for this type of problems.
We derive of the base classes `Problem` and `Solution` available in discrete-optimization:

```python
from discrete_optimization.generic_tools.do_problem import Problem, Solution


class MyKnapsackProblem(Problem): ...


class MyKnapsackSolution(Solution): ...

```

Then use your favorite IDE to generate the methods to implement (or look in the source file of the base classes for abtract methods).
You should get something like:

```python
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ObjectiveRegister,
    Problem,
    Solution,
)


class MyKnapsackProblem(Problem):

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def satisfy(self, variable: Solution) -> bool:
        pass

    def get_attribute_register(self) -> EncodingRegister:
        pass

    def get_solution_type(self) -> type[Solution]:
        pass

    def get_objective_register(self) -> ObjectiveRegister:
        pass


class MyKnapsackSolution(Solution):

    def copy(self) -> Solution:
        pass

```

:::{tip}
- with PyCharm, right-click on the class, and select "Generate..." > "Implement Methods..."
- with VSCode, click on the class name, then on the lightbulb that comes up above, then "Implement all inherited abstract classes"
:::


## Implementation

We start by implementing a first part of `MyKnapsackProblem`:
- constructor: set the main characteristics of the instance (items and max capacity)
- get_solution_type: link to `MyKnapsackSolution`
- get_objective_register: document which objectives can be computed
  (and are keys of the dictionary returned by `evaluate()`), and how to aggregate them.
  See {py:obj}`discrete_optimization.generic_tools.do_problem.ObjectiveRegister` for more information.

```python
Item = tuple[int, int]  # value, weight


class MyKnapsackProblem(Problem):
    def __init__(self, items: list[Item], max_capacity: int):
        self.items = items
        self.max_capacity = max_capacity

    def get_solution_type(self) -> type[Solution]:
        """Specify associated solution type."""
        return MyKnapsackSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Specify the different objectives and if we need to aggregate them."""
        return ObjectiveRegister(
            dict_objective_to_doc=dict(
                # total value of taken items: main objective
                value=ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
                # weight violation (how much we exceed the max capactity): penalty to be removed with a big coefficient
                weight_violation=ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-1000.0
                ),
            ),
            objective_handling=ObjectiveHandling.AGGREGATE,  # aggregate both objective
            objective_sense=ModeOptim.MAXIMIZATION,  # maximize resulting objective
        )

```

Before going on, we will implement `MyKnapsackSolution` as its characteristics will be needed for the remaining methods of `MyKnapsackProblem`.
Several notes:
- The base class Solution already implements a `__init__()` method that stores the problem the solution is related to.
  We call it in our constructor and also specify the `problem` attribute type so that the IDE can type it properly.
- We implement also `lazy_copy()` which defaults to `copy()` but here avoid a deep copy of `list_taken` to improve
  the performance of evolutionary algorithms that mutate the solutions.


```python
class MyKnapsackSolution(Solution):
    """Solution class for MyKnapsackProblem.

    Args:
        problem: problem instance for which this is a solution
        list_taken: list of booleans specifying if corresponding item has been taken.
            Must be of same length as problem.items

    """

    problem: MyKnapsackProblem  # help the IDE to type correctly

    def __init__(self, problem: MyKnapsackProblem, list_taken: list[bool]):
        super().__init__(problem=problem)  # stores the problem attribute
        self.list_taken = list_taken

    def copy(self) -> Solution:
        """Deep copy the solution."""
        return MyKnapsackSolution(
            problem=self.problem, list_taken=list(self.list_taken)
        )

    def lazy_copy(self) -> Solution:
        """Shallow copy the solution.

        Not mandatory to implement but can increase the speed of evolutionary algorithms.

        """
        return MyKnapsackSolution(problem=self.problem, list_taken=self.list_taken)

```

Now we can finish the implementation of the problem:

```python
class MyKnapsackProblem(Problem):

    ...

    def get_attribute_register(self) -> EncodingRegister:
        """Describe attributes of a solution.

        To be used by evolutionary solvers to choose the appropriate mutations
        without implementing a dedicated one.

        """
        return EncodingRegister(
            dict_register={
                "taken": {
                    "name": "list_taken",
                    "type": [TypeAttribute.LIST_BOOLEAN],
                    "n": len(self.items),
                }
            }
        )

    def evaluate(self, variable: Solution) -> dict[str, float]:
        """Evaluate the objectives corresponding to the solution.

        The objectives must match the ones defined in `get_objective_register`.

        """
        if not isinstance(variable, MyKnapsackSolution):
            raise ValueError("variable must be a `MyKnapsackSolution`")
        value = self.compute_total_value(variable)
        weight = self.compute_total_weight(variable)
        return dict(value=value, weight_violation=max(0, weight - self.max_capacity))

    def satisfy(self, variable: Solution) -> bool:
        """Check that the solution satisfies the problem.

        Check the weight violation.

        """
        if not isinstance(variable, MyKnapsackSolution):
            return False
        return self.compute_total_weight(variable) <= self.max_capacity

    def compute_total_weight(self, variable: MyKnapsackSolution) -> int:
        """Compute the total weight of taken items."""
        return sum(
            taken * weight
            for taken, (value, weight) in zip(variable.list_taken, self.items)
        )

    def compute_total_value(self, variable: MyKnapsackSolution) -> int:
        """Compute the total value of taken items."""
        return sum(
            taken * value
            for taken, (value, weight) in zip(variable.list_taken, self.items)
        )
```

The complete resulting python module with creation of problem, solutions and evaluation and satifiyability checks can be found here: <path:tutorial_new_problem.py>.

## Apply a generic solver

Now that we have implemented a new problem, we can try to solve it with an existing solver.
In discrete-optimization, most solvers are specialized to a problem class.
But evolutionary algorithms (from `discrete_optimization.generic_tools.ea`) or local search (from `discrete_optimization.generic_tools.ls`) can be applied directly.

In this section we use simulated annealing to solve a knapsack instance.

We need to define an instance of knapsack problem. We also define a dummy solution needed
to be used as starting point.

```python
# instantiate a knapsack problem
problem = MyKnapsackProblem(
    max_capacity=10,
    items=[
        (2, 5),  # item 0: value=2, weight=5
        (3, 1),  # item 1: value=3, weight=1
        (2, 4),  # item 2: value=2, weight=4
        (5, 9),  # item 3: value=5, weight=9
    ]
)
# dummy solution (not taking anything)
solution = MyKnapsackSolution(
    problem=problem,
    list_taken=[False,] * len(problem.items)
)
```

The simulated annealing will need a mutation to apply at each step.
We can select all mutations compatible with the declared solution attributes.

```python
mixed_mutation = create_mixed_mutation_from_problem_and_solution(
    problem=problem,
    solution=solution,
)
```

Then we create the simulated annealing solver:
```python
# restart and temperature handler
restart_handler = RestartHandlerLimit(3000)
temperature_handler=TemperatureSchedulingFactor(1000, restart_handler, 0.99)

# simulated annealing solver
sa = SimulatedAnnealing(
        problem=problem,
        mutator=mixed_mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
    )
```

And solve, get the best solution, display it, and check its satisfiability:
```python
result_storage = sa.solve(
    initial_variable=solution,
    nb_iteration_max=1000,  # increase for a more realistic problem instance
)

sol, fit = result_storage.get_best_solution_fit()

items_taken_indices = [i for i, taken in enumerate(sol.list_taken) if taken]

print(f"Best fitness: {fit}")
print(f"Taking items n°: {items_taken_indices}")

assert problem.satisfy(sol)
```

The complete python script can be found here: <path:tutorial_new_problem_ls.py>,
and should be run near the previous module <path:tutorial_new_problem.py> so that you can import your new classes.


## To go further

### Tasks problem (scheduling/allocation)

If your problem can be seen as an allocation or a scheduling problem,
you should consider also implementing the related API
(i.e. derive from `AllocationProblem` or `SchedulingProblem` from `discrete_optimization.generic_tasks_tools` subpackage).

If you do so:
- you will soon have access to an autogenerated cpsat solver (work in progress);
- you will have access to a dedicated constraint handler working with any cp solver implementing the appropriate API
  (i.e. deriving from `AllocationSolver` or `SchedulingSolver`), to go with a generic LNS (Large Neighborhood Search) solver.

See the [dedicated tutorial](../tasks_problem/tutorial_tasks_problem.md).
