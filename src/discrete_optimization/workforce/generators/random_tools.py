# Using code  https://github.com/TUPLES-Trustworthy-AI/beluga-challenge-tools-internal/blob/97aeafcb75c3e570a8b4c1d38a13ef345d8cc636/generator/configurations/random_state.py

from typing import TypeVar

import numpy as np
from numpy.random import Generator
from scipy.stats import truncnorm, uniform

epsilon = 0.00001


class RandomState:
    def __init__(self, seed) -> None:
        self.rng: Generator = np.random.default_rng(seed)

    T = TypeVar("T")

    def get_random_element_uniform(self, list: list[T]) -> T:
        assert len(list) > 0
        return list[self.get_discrete_truncated_uniform_sample(0, len(list) - 1)]

    def get_random_element_prop(self, list: list[T], probs: list[float]) -> T:
        assert len(list) > 0
        assert len(list) == len(probs)
        sum_props = sum(probs)
        assert 1 - epsilon <= sum_props <= 1 + epsilon, (
            f"Sum of probabilities must be 1 +-{epsilon} but is {sum(probs)}"
        )

        sample = uniform.rvs()
        index = 0
        ref = 0
        for next_prob in probs:
            ref += next_prob
            if sample <= ref:
                return list[index]

            index += 1

        assert False

    def get_discrete_truncated_uniform_sample(self, lower: int, upper: int) -> int:
        return int(round(uniform.rvs(lower, upper - lower, random_state=self.rng)))

    def get_discrete_truncated_normal_sample(
        self, center: int, sigma: int, lower: int, upper: int
    ) -> int:
        if lower == upper:
            return lower
        low = (lower - center) / sigma
        up = (upper - center) / sigma
        r = truncnorm.rvs(low, up, loc=center, scale=sigma, random_state=self.rng)
        return int(round(r))

    def get_uniform_advantage_sample(self, advantage: list[int]) -> int:
        distribution = []

        if sum(advantage) > 0:
            for i, a in enumerate(advantage):
                distribution += [i for _ in range(a)]
        else:
            distribution += [i for i in range(len(advantage))]

        return self.get_random_element_uniform(distribution)
