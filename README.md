# Discrete Optimization

Discrete Optimization is a python library to ease the definition and re-use of discrete optimization problems and solvers. It is implemented by the CRT AI Research as part of the IOODA project.

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

In addition, the library contains functionalities to enable robust optimization through different scenario handling mechanisms) and multi-objective optimization (aggregation of objectives, Pareto optimization, MO post-processing).


## Installation 

- Install minizinc (https://www.minizinc.org/)
- Install discrete-optimisation
```
git clone git@github.airbus.corp:Airbus/discrete-optimisation.git
cd discrete-optimisation
pip install --editable .
```

### Installation with conda : 
- Install minizinc (https://www.minizinc.org/)
- Install discrete-optimisation using conda
```
conda create -n do_lib_env python=3.8
conda activate do_lib_env
pip install .
```
Install gurobi python binding if you want to use gurobi in some solver. (the gurobi lib in pip is not the good one)
```
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi
```
### Problem with installation ?
Please push an issue in the repo so that any issue can be studied.

## Example scripts
We recommend browsing through the `tests` and `tests_clean` folder to see examples on how to use the different features of 
the library and how to instantiate different problem instances and solvers. For a more detailed overview, we recommend looking at the scheduling notebooks.

## Routing examples :
see the notebooks `tests_clean/pickup_vrp/*.ipynb` or `tests_clean/pickup_vrp/loading_example.py` 

## Example notebooks :
In the `notebooks` directory, you will find several jupyter notebooks demonstrating how the library can be used on a scheduling problem (RCPSP). The notebooks are well commented, highlighting different approaches and demonstrating the use of different solvers.

## Unit tests : 
if you install `pytest` library you can then run `pytest -v`  in tests_clean/ folder or any subfolder. Unit test are not constantly checked, so 
