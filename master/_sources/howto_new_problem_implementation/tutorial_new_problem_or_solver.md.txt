# How to write its own problem/solver class?




```{contents}
---
depth: 5
local: true
---
```

## About discrete-optimization key concepts

The discrete-optimization library introduce 3 main classes:
- `Problem`: describes a discrete optimization problem, its objectives, how to evaluate a solution, how to check its validity
- `Solution`: a thin wrapper around the numeric attributes (numpy array, list of integers, ...) defining a solution to the problem
- `SolverDO`: base class for all solvers of the library

The following diagram shows the interactions between those classes:

![diagram](do-concepts-diagram.png)


The main benefits of having such a common API are:
- Insuring fair comparison between solvers for a given problem
- Capitalizing solvers and models
- Combining solvers in meta-solvers
- Benchmark and visualization via a [dashboard](../dashboard)
- Hyperparameters optimization



## Brief presentation of knapsack problem

In this tutorial, we take the example of the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) (because of its simplicity).

This problem and adapted solvers have already been implemented in the library (see `discrete_optimization.knapsack` package)
and a [dedicated tutorial](../notebooks.md#knapsack-problem) on how to use them is available in Notebooks section.

We focus here on how we could write them from scratch.

The knapsack problem is a very common combinatorial optimization problem where you are given a knapsack of a given weight capacity $C$
and a bunch of items with values and weights.
The goal is to fill the knapsack with the best aggregated value, respecting the maximum weight constraint.

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

When implementing a solver based on another existing optimization library like OR-Tools or Gurobi,
discrete-optimization have already some wrappers prepared  for you.

In these wrappers, the `solve()` method is already implemented, taking into account the main parameters from the 3rd party library,
handling the callbacks and sometimes already managing other extra-features like warm-start or [explainability](../notebooks.md#explaining-unsatisfiability).

Generally, you will just have to implement:
- `init_model()` that translates the problem in the other library language,
- `retrieve_solution()` or equivalent, in charge of translating solutions in d-o format.

In the next section, we show how to use the OR-Tools/CP-SAT and OR-Tools/MathOpt wrappers. A curated list of other wrappers
is available in the ["To go further"](#list-of-wrappers) section.


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
