#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from collections import namedtuple
from typing import Tuple

import networkx as nx
import numpy as np

from discrete_optimization.vrp.vrp_model import VrpProblem

logger = logging.getLogger(__name__)


def compute_length_matrix(vrp_model: VrpProblem) -> Tuple[np.ndarray, np.ndarray]:
    nb_customers = vrp_model.customer_count
    matrix_distance = np.zeros((nb_customers, nb_customers))
    for f in range(nb_customers):
        for c in range(f + 1, nb_customers):
            matrix_distance[f, c] = vrp_model.evaluate_function_indexes(f, c)
            matrix_distance[c, f] = matrix_distance[f, c]
    closest = np.argsort(matrix_distance, axis=1)
    return closest, matrix_distance


def prune_search_space(
    vrp_model: VrpProblem, n_shortest: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    closest, matrix_distance = compute_length_matrix(vrp_model)
    matrix_adjacency = np.zeros(matrix_distance.shape, dtype=np.int_)
    nb_customers = vrp_model.customer_count
    if n_shortest < nb_customers:
        for c in range(matrix_adjacency.shape[0]):
            matrix_adjacency[c, closest[c, :n_shortest]] = matrix_distance[
                c, closest[c, :n_shortest]
            ]
            matrix_adjacency[c, 0] = matrix_distance[c, 0]
    else:
        matrix_adjacency = matrix_distance
    return matrix_adjacency, matrix_distance


def build_graph(vrp_model: VrpProblem) -> Tuple[nx.Graph, np.ndarray]:
    matrix_adjacency, matrix_distance = prune_search_space(
        vrp_model=vrp_model, n_shortest=vrp_model.customer_count
    )
    G = nx.from_numpy_matrix(matrix_adjacency, create_using=nx.DiGraph)
    G.add_edge(0, 0, weight=0)
    return G, matrix_distance
