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

Quick version:
```shell
pip install discrete-optimization
```
For more details, see the [online documentation](https://airbus.github.io/discrete-optimization/master/install).

## Documentation

The latest documentation is available [online](https://airbus.github.io/discrete-optimization).

## Examples

Some educational notebooks are available in `notebooks/` folder.
Links to launch them online with [binder](https://mybinder.org/) are provided in the
[Notebooks section](https://airbus.github.io/discrete-optimization/master/notebooks) of the online documentation.

More examples can be found as Python scripts in the `examples/` folder, using the different features of
the library and showing how to instantiate different problem instances and solvers.

## Contributing

See more about how to contribute in the [online documentation](https://airbus.github.io/discrete-optimization/master/contribute).


## License

This software is under the MIT License that can be found in the [LICENSE](./LICENSE) file at the root of the repository.

Some minzinc models have been adapted from files coming from
- https://github.com/MiniZinc/minizinc-benchmarks under the same [license](https://github.com/MiniZinc/minizinc-benchmarks/blob/master/LICENSE),
- https://github.com/youngkd/MSPSP-InstLib for which we have the written authorization of the author.
