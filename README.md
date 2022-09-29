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

In addition, the library contains functionalities to enable robust optimization
through different scenario handling mechanisms) and multi-objective optimization
(aggregation of objectives, Pareto optimization, MO post-processing).


## Installation

### Prerequisites

- Install [minizinc](https://www.minizinc.org/), at least version 2.6, and update the `PATH` environment variable
so that it can be found by Python. See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.
- Optionally, install [gurobi](https://www.gurobi.com/) with its python binding (gurobipy)
  and an appropriate license, if you want to try solvers that make use of gurobi.

  NB: If you just do `pip install gurobipy`, you get a minimal license which does not allow to use it on "real" models.

### Normal install

Install discrete-optimization from pip:

```shell
pip install discrete-optimization
```

### Install in developer mode

You can also install the library directly from the repository in developer mode:

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

In the `notebooks` directory of the repository, you will find several jupyter notebooks demonstrating
how the library can be used
- on a knapsack problem,
- on a scheduling problem (RCPSP).


### Scripts

The `examples` directory of the repository gather several scripts using the different features of
the library and how to instantiate different problem instances and solvers.


## Unit tests

Unit tests are available in `tests/` directory of the repository.
To test the library, you can install the library
with the "test" extra dependencies by typing
```shell
git clone https://github.com/airbus/discrete-optimization.git
cd discrete-optimization
pip install --editable .[test]
```


Then run pytest on tests folder:
```shell
pytest -v tests
```


## License

This software is under the MIT License that can be found in the [LICENSE](./LICENSE) file at the root of the repository.

Some minzinc models have been adapted from files coming from
- https://github.com/MiniZinc/minizinc-benchmarks under the same [license](https://github.com/MiniZinc/minizinc-benchmarks/blob/master/LICENSE),
- https://github.com/youngkd/MSPSP-InstLib for which we have the written authorization of the author.
