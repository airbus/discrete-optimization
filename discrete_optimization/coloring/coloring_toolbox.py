#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Hashable, List, Optional, Tuple

import networkx as nx


def compute_cliques(
    g: nx.Graph, nb_max: Optional[int] = None
) -> Tuple[List[List[Hashable]], bool]:
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
