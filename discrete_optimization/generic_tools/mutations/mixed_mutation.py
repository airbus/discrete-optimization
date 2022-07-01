import os
from typing import List, Union

import numpy as np

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import Solution


class BasicPortfolioMutation(Mutation):
    def __init__(
        self,
        list_mutation: List[Mutation],
        weight_mutation: Union[List[float], np.array],
    ):
        self.list_mutation = list_mutation
        self.weight_mutation = weight_mutation
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation / np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutation)), dtype=np.int)
        print(len(self.list_mutation), " mutation available")

    def mutate(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate(solution)

    def mutate_and_compute_obj(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate_and_compute_obj(solution)


class BasicPortfolioMutationTrack(Mutation):
    def __init__(
        self,
        list_mutation: List[Mutation],
        weight_mutation: Union[List[float], np.array],
    ):
        self.list_mutation = list_mutation
        self.weight_mutation = weight_mutation
        if isinstance(self.weight_mutation, list):
            self.weight_mutation = np.array(self.weight_mutation)
        self.weight_mutation = self.weight_mutation / np.sum(self.weight_mutation)
        self.index_np = np.array(range(len(self.list_mutation)), dtype=np.int)

    def mutate(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate(solution)

    def mutate_and_compute_obj(self, solution: Solution):
        choice = np.random.choice(self.index_np, size=1, p=self.weight_mutation)[0]
        return self.list_mutation[choice].mutate_and_compute_obj(solution)
