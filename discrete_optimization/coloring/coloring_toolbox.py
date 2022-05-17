import networkx as nx
from typing import Tuple, List, Hashable


def compute_cliques(g: nx.Graph, nb_max=None) -> Tuple[List[List[Hashable]], bool]:
    """Compute nb_max cliques for a given graph (possibly a graph from a coloring model).

    A clique of a graph is a subset of nodes that are all adjacent to each other.
    This is quite relevant for coloring problem where the color of nodes of a clique
    will have to be different.

    Params:
        g: a network x Graph
        nb_max: max number of cliques to return.

    Returns: A list of cliques and a boolean indicating if all cliques have been computed.

    """
    cliques = []
    nb = 0
    not_all = False
    for c in nx.algorithms.clique.find_cliques(g):
        cliques += [c]
        nb += 1
        if nb_max is not None and nb >= nb_max:
            not_all = True
            break
    return cliques, not_all
