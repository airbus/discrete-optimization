# External solvers hub.
This module is aimed to store some wrapper to solver with low integration level with the lib,
but that are relevant to some of our problem.
As this is not a mandatory library that comes from the lib, we build it outside the classical way (with cp_tools, lp_tools, and so on).


## Exhaustive list of external solvers partially wrapped to discrete-opt, and associated problems:

| Solver                                                        | Description                              | Problems                     |
|---------------------------------------------------------------|------------------------------------------|------------------------------|
| [Tempo](https://gepgitlab.laas.fr/roc/emmanuel-hebrard/tempo) | Disjunctive scheduling solver            | Workforce scheduling problem |
| KAMIS                                                         | Local search methods for independent set | Maximum independent set      |
| LKH (planned)                                                 | Heuristic for TSP and variants           | TSP/VRP                      |
|
For tempo, as we will use it in several problem, we create some common API in this module.
