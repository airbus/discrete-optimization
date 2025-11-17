# How to write its own problem/solver class?

In this tutorial, we take the example of the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) (because of its simplicity).

This problem and adapted solvers have already been implemented in the library (see `discrete_optimization.knapsack` package)
and a [dedicated tutorial](../notebooks.md#knapsack-problem) on how to use them is available in Notebooks section.

We focus here on how we could write them from scratch.


```{contents}
---
depth: 5
local: true
---
```


## Brief presentation of knapsack problem

This is a very common combinatorial optimization problem where you are given a knapsack of a given weight capacity $C$
and a bunch of items with values and weight.
The goal is to fill the knapsack with the best aggregated value, respecting the weight constraint.

![knapsack problem illustration](https://upload.wikimedia.org/wikipedia/commons/f/fd/Knapsack.svg "Image from wikipedia: https://commons.wikimedia.org/wiki/File:Knapsack.svg").

We handle here the *0-1 knapsack problem* where each item can only be taken once.


## How to write its own problem class?

```{include} tutorial_new_problem.md
:heading-offset: 1
:start-line: 1
```

## How to write its own solver class?

Now that we have a new problem and associated solution classes `MyKnapsackProblem`/`MyKnapsackSolution`,
let us create solvers adapted to it.

### Inheriting directly from base class `SolverDO` (ex: the greedy solver)

```{include} tutorial_new_solver_greedy.md
:heading-offset: 2
:start-line: 1
```

### Taking advantage of d-o wrappers for 3rd-party libraries

When implementing a solver based on another existing optimization library like ortools or gurobi,
discrete-optimization have already some wrappers prepared  for you
(see a list of such wrappers [below](#list-of-wrappers)).

In that case, the `solve()` method is already implemented, taking into account the main parameters from the 3rd party library,
handling the callbacks and sometimes already managing other stuff like warm-start or [explainability](../notebooks.md#explaining-unsatisfiability)
(the latter only available for the cpmpy wrapper).

Generally, you will just have to implement:
- `init_model()` that translates the problem in the other library language,
- `retrieve_solution()` or equivalent, in charge of translating solutions in d-o format.


#### CP solver  (ex: OR-Tools/CP-SAT)

```{include} tutorial_new_solver_cpsat.md
:heading-offset: 3
:start-line: 1
```

#### MILP solver (ex: OR-Tools/MathOpt)

```{include} tutorial_new_solver_mathopt.md
:heading-offset: 3
:start-line: 1
```

### To go further

```{include} tutorial_new_solver_go_further.md
:heading-offset: 2
:start-line: 1
```
