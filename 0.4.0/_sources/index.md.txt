# Welcome to discrete-optimization's documentation!

Discrete Optimization is a python library to ease the definition and re-use of discrete optimization problems and solvers.
It has been initially developed in the frame of [scikit-decide](https://github.com/airbus/scikit-decide) for scheduling.
The code base starting to be big, the repository has now been splitted in two separate ones.

The library contains a range of existing solvers already implemented such as:
* greedy methods
* local search (Hill Climber, Simulated Annealing)
* metaheuristics (Genetic Algorithms, NSGA)
* linear programming
* constraint programming
* hybrid methods (LNS)

The library also contains implementation of several classic discrete optimization problems:
* Travelling Salesman Problem (TSP)
* Knapsack Problem (KP)
* Vehicle Routing Problem (VRP)
* Facility Location Problem (FLP)
* Resource Constrained Project Scheduling Problem (RCPSP). Several variants of RCPSP are available
* Graph Colouring Problem (GCP)

In addition, the library contains functionalities to enable robust optimization
through different scenario handling mechanisms) and multi-objective optimization
(aggregation of objectives, Pareto optimization, MO post-processing).


```{toctree}
---
maxdepth: 2
caption: Contents
---
install
getting_started
notebooks
api/modules
contribute
Github  <https://github.com/airbus/discrete-optimization>
```
