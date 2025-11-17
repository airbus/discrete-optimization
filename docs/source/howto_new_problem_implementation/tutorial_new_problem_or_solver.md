# How to write its own problem/solver class?

In this tutorial, we take the example of the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) (because of its simplicity).

This problem and adapted solvers have already been implemented in the library (see `discrete_optimization.knapsack` package)
and a [dedicated tutorial](notebooks.md#knapsack-problem) on how to use them is available in Notebooks section.

We focus here on how we could write them from scratch.


```{contents}
---
depth: 3
local: true
---
```


## Brief presentation of knapsack problem

This is a very common combinatorial optimization problem where you are given a knapsack of a given weight capacity $C$
and a bunch of items with values and weight.
The goal is to fill the knapsack with the best aggregated value, respecting the weight constraint.

![knapsack problem illustration](https://upload.wikimedia.org/wikipedia/commons/f/fd/Knapsack.svg "Image from wikipedia: https://commons.wikimedia.org/wiki/File:Knapsack.svg").

We handle here the *0-1 knapsack problem* where each item can only be taken once.


[//]: # (How to write its own problem?)
```{include} tutorial_new_problem.md
:heading-offset: 1
```


[//]: # (How to write its own solver?)
```{include} tutorial_new_solver.md
:heading-offset: 1
```
