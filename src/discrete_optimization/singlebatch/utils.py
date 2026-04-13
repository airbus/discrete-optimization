import random
from enum import Enum
from typing import Tuple

from discrete_optimization.singlebatch.problem import Job, SingleBatchProcessingProblem


class GenerationMode(Enum):
    """Modes for generating job sizes and processing times."""

    RANDOM = "random"
    POSITIVE_CORRELATION = "positive"
    NEGATIVE_CORRELATION = "negative"


def generate_random_batch_problem(
    nb_jobs: int = 50,
    capacity: int = 100,
    size_range: Tuple[int, int] = (10, 50),
    duration_range: Tuple[int, int] = (5, 100),
    mode: GenerationMode = GenerationMode.RANDOM,
    noise_level: float = 0.2,
    seed: int = 42,
) -> SingleBatchProcessingProblem:
    """
    Generates a SingleBatchProcessingProblem instance.

    Args:
        nb_jobs: Number of jobs to generate.
        capacity: Maximum capacity of the batch processing machine.
        size_range: Tuple (min_size, max_size) for jobs.
        duration_range: Tuple (min_duration, max_duration) for jobs.
        mode: GenerationMode defining the correlation between size and duration.
        noise_level: Float between 0 and 1 defining how much randomness to add to correlated modes.
        seed: Random seed for reproducibility.

    Returns:
        A populated SingleBatchProcessingProblem.
    """
    random.seed(seed)
    jobs = []

    min_s, max_s = size_range
    min_d, max_d = duration_range

    for i in range(nb_jobs):
        duration = None
        if mode == GenerationMode.RANDOM:
            # Totally independent variables
            size = random.randint(min_s, max_s)
            duration = random.randint(min_d, max_d)

        elif mode == GenerationMode.POSITIVE_CORRELATION:
            # Larger sizes tend to have longer processing times
            factor = random.random()
            size = int(min_s + factor * (max_s - min_s))

            base_dur = min_d + factor * (max_d - min_d)
            noise = random.uniform(-noise_level, noise_level) * (max_d - min_d)
            duration = int(max(min_d, min(max_d, base_dur + noise)))

        elif mode == GenerationMode.NEGATIVE_CORRELATION:
            # Larger sizes tend to have shorter processing times
            factor = random.random()
            size = int(min_s + factor * (max_s - min_s))

            base_dur = max_d - factor * (max_d - min_d)
            noise = random.uniform(-noise_level, noise_level) * (max_d - min_d)
            duration = int(max(min_d, min(max_d, base_dur + noise)))

        jobs.append(Job(job_id=i, processing_time=duration, size=size))

    return SingleBatchProcessingProblem(jobs=jobs, capacity=capacity)
