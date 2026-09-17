# CP-SAT solver

Let us see how to implement a constraint programming solver making use of the [OR-Tools/CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver).


## Creating the skeleton

We start by deriving from our wrapper `OrtoolsCpSatSolver`:

```python
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class MyKnapsackCpSatSolver(OrtoolsCpSatSolver): ...

```

Then, we generate the methods to implement (for instance thanks to a smart IDE or by looking at `OrtoolsCpSatSolver` source code).
We also override `init_model()` (it is not in abstract method as it is already trivially implemented in base class `SolverDO`).
You should get something like:

```python
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class MyKnapsackCpSatSolver(OrtoolsCpSatSolver):
    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        pass

```

:::{tip}
- with PyCharm, right-click on the class, and select "Generate..." > "Implement Methods..." / "Override Methods..."
- with VSCode, click on the class name, then on the lightbulb that comes up above, then "Implement all inherited abstract classes".
  Then start by writing `def init_model()`, it should complete it for you.
:::

## Implementation

- `__init__()`: this method is already defined by parent class, and sets the attribute `problem`.
  To help the IDE to type correctly, we can specify its expected class.
- `init_model()`: the method from the super class initializes a `cp_model` attribute of type `ortools.sat.python.cp_model.CpModel`
  in which we must encode our knapsack problem.
- `retrieve_solution()`: we must translate the internal solution of the CP-SAT solver into a `MyKnapsackSolution`.
  This will be used for each new solution found via an ortools callback.



```python
class MyKnapsackCpSatSolver(OrtoolsCpSatSolver):
    """CP-SAT solver for the knapsack problem."""

    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def init_model(self, **kwargs: Any) -> None:
        """Init the CP model."""
        super().init_model(**kwargs)  # initialize self.cp_model
        # create the boolean variables for each item
        self.variables = [self.cp_model.new_bool_var(name=f"x_{i}") for i in range(len(self.problem.items))]
        # add weight constraint
        total_weight = sum(
            self.variables[i] * weight
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.cp_model.add(total_weight <= self.problem.max_capacity)
        # maximize value
        total_value =  sum(
            self.variables[i] * value
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.cp_model.maximize(total_value)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        """Translate a cpsat solution into a d-o solution.

        Args:
            cpsolvercb:  the ortools callback called when the cpsat solver finds a new solution.

        Returns:

        """
        taken = [bool(cpsolvercb.Value(var)) for var in self.variables]
        return MyKnapsackSolution(problem=self.problem, list_taken=taken)
```

We can solve the problem with a callback logging the objective at each step via:
```python
solver = MyKnapsackCpSatSolver(problem=problem)
solution = solver.solve(callbacks=[ObjectiveLogger()]).get_best_solution()
```

## Solver in action

The code for this CP-SAT solver and how to use it can be found here: <path:tutorial_new_solver_cpsat.py>.
Note that it should be run in the same directory as the previous module
<path:tutorial_new_problem.py> that declares the knapsack problem and solution classes, so that they can be imported.
