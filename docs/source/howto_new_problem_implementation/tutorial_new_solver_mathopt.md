# MathOpt solver

Let us see how to implement a mixed integer linear programming solver making use of the [OR-Tools/MathOpt solver](https://developers.google.com/optimization/math_opt).


## Creating the skeleton

We start by deriving from our wrapper `OrtoolsMathOptMilpSolver`:

```python
from discrete_optimization.generic_tools.lp_tools import OrtoolsMathOptMilpSolver


class MyKnapsackMathOptSolver(OrtoolsMathOptMilpSolver): ...

```

Then, we generate the methods to implement (only the fully necessary ones), which are `init_model()`
and `retrieve_solution()`:

```python
from typing import Any, Callable

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.lp_tools import OrtoolsMathOptMilpSolver


class MyKnapsackMathOptSolver(OrtoolsMathOptMilpSolver):
    def init_model(self, **kwargs: Any) -> None:
        pass

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        pass

```

:::{tip}
- with PyCharm, right-click on the class, and select "Generate..." > "Implement Methods..." / "Override Methods..."
- with VSCode, click on the class name, then on the lightbulb that comes up above, then "Implement all inherited abstract classes".
  Then start by writing `def init_model()`, it should complete it for you.
:::

## Implementation

- `__init__()`: this method is already defined by parent class, and set the attribute `problem`.
  To help the IDE to type correctly, we can specify its expected class.
- `init_model()`: the method must initialize a `model` attribute of type `ortools.math_opt.python.model.Model`
  in which we must encode our knapsack problem.
  - `retrieve_solution()`: we must translate the internal solution of the mathopt solver into a `MyKnapsackSolution`.
    As this is a method of the more generic `MilpSolver` class, it takes callables as argument
    that are responsible for mapping variable into their value in internal solution. This callable will be automatically
    adapted to the mathopt framework (and updated accordingly in gurobi wrapper)
    Note that the value can be a float number between 0 and 1, so we check whether it is above or below 0.5.


```python
class MyKnapsackMathOptSolver(OrtoolsMathOptMilpSolver):
    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def init_model(self, **kwargs: Any) -> None:
        """Create mathopt `model` to encode the knapsack problem."""
        self.model = mathopt.Model()
        self.variables = [
            self.model.add_binary_variable(name=f"x_{i}")
            for i in range(len(self.problem.items))
        ]
        # add weight constraint
        total_weight = mathopt.LinearSum(
            self.variables[i] * weight
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.model.add_linear_constraint(total_weight <= self.problem.max_capacity)
        # maximize value
        total_value = mathopt.LinearSum(
            self.variables[i] * value
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.model.maximize(total_value)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """Translate the mathopt solution into a d-o solution

        Args:
            get_var_value_for_current_solution: mapping a mathopt variable to its value in the solution
            get_obj_value_for_current_solution: returning the mathopt objective value

        Returns:

        """
        return MyKnapsackSolution(
            problem=self.problem,
            list_taken=[
                get_var_value_for_current_solution(var)
                >= 0.5  # represented by a float between 0. and 1.
                for var in self.variables
            ],
        )
```

We can solve the problem with a callback logging the objective at each step via:
```python
solver = MyKnapsackCpSatSolver(problem=problem)
solution = solver.solve(
    mathopt_solver_type=mathopt.SolverType.GSCIP,  # choose your preferred mathopt subsolver
    callbacks=[ObjectiveLogger()]
).get_best_solution()
```

### Warm start

Moreover, if we implement the method `convert_to_variable_values()` that translates a solution into
a mapping variable -> value, we enable warmstart:

```python
class MyKnapsackMathOptSolver(OrtoolsMathOptMilpSolver):

    ...

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        assert isinstance(solution, MyKnapsackSolution)
        return {
            var: float(taken) for var, taken in zip(self.variables, solution.list_taken)
        }


solver = MyKnapsackCpSatSolver(problem=problem)
solver.init_model()  # explicit call to init_model() needed to make work warm-start
# warm-start to a specified solution
solver.set_warm_start(a_previous_solution)
# solve will start from it (depends on the mathopt subsolver chosen though)
solution = solver.solve(
    mathopt_solver_type=mathopt.SolverType.GSCIP,
    callbacks=[ObjectiveLogger()]
).get_best_solution()
```


## Solver in action

The code for this mathopt solver and how to use it can be found here: <path:tutorial_new_solver_mathopt.py>.
Note that it should be run near the previous module
<path:tutorial_new_problem.py> that declares the knapsack problem and solution classes, so that they can be imported.


:::{Note}
The milp wrappers in d-o (for now mathopt and gurobi) share a common API to define a model, so that it is easy to switch from one to another.
Here, for simplicity we chose to directly use the mathopt API, but we could also have implemented `init_model()` in a common base class like:
```python
class _BaseKnapsackMilpSolver(MilpSolver):
    def init_model(self, **kwargs: Any) -> None:
        self.model = self.create_empty_model()
        self.variables = [
                    self.add_binary_variable(name=f"x_{i}")
                    for i in range(len(self.problem.items))
                ]
        total_weight = self.construct_linear_sum(
            self.variables[i] * weight
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.add_linear_constraint(total_weight <= self.problem.max_capacity)
        total_value = self.construct_linear_sum(
            self.variables[i] * value
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.set_model_objective(total_value, minimize=False)
```
then inherit it with `OrtoolsMathOptMilpSolver` or `GurobiMilpSolver` to generate a mathopt or gurobi knapsack solver.

This is how it is done in the d-o implementation of knapsack milp solver.
See [`discrete_optimization.knapsack.solvers.lp`](https://github.com/airbus/discrete-optimization/blob/master/src/discrete_optimization/knapsack/solvers/lp.py) for more details.

:::
