# Discrete Optimization

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
* Travelling thief Problem (TTP)

In addition, the library contains functionalities to enable robust optimization
through different scenario handling mechanisms) and multi-objective optimization
(aggregation of objectives, Pareto optimization, MO post-processing).


## Installation

- Install [minizinc](https://www.minizinc.org/).
- Optionally, install [gurobi](https://www.gurobi.com/) with its python binding (gurobipy)
  and an appropriate license, if you want to try solvers that make use of gurobi.

  NB: If you just do `pip install gurobipy`, you get a minimal license which does not allow to use it on "real" models.
- Install discrete-optimization:
    ```shell
    git clone https://github.com/airbus/discrete-optimization.git
    cd discrete-optimization
    pip install --editable .
    ```


If you encounter any problem during installation,
please fill an [issue](https://github.com/airbus/discrete-optimization/issues)
on the repository.


## Examples

### Notebooks

*[work in progress]*

In the `notebooks` directory, you will find several jupyter notebooks demonstrating
how the library can be used on a scheduling problem (RCPSP).

[//]: # (The notebooks are well commented, highlighting different approaches
and demonstrating the use of different solvers.)

### Scripts

The `examples` directory gather several scripts using the different features of
the library and how to instantiate different problem instances and solvers.


## Unit tests

To test the library, you can install the library with the additional necessaries dependencies with
```shell
pip install --editable .[test]
```
Then run pytest on tests folder:
```shell
pytest -vv tests
```
