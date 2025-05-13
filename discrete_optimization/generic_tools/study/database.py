from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from types import TracebackType
from typing import TYPE_CHECKING, Optional

import pandas as pd

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study.experiment import (
    CONFIG_PREFIX,
    ID,
    INSTANCE,
    METRICS,
    NAME,
    PARAMETERS,
    REASON,
    SOLVER,
    STATUS,
    Experiment,
    SolverJsonableConfig,
)

if TYPE_CHECKING:
    from pandas._typing import Self

logger = logging.getLogger(__name__)


class Database(ABC):
    """Base class for database storing experiments.

    By default, we assume a database is associated with a given study.
    But it could be implemented so that it can store several studies at once.

    d-o experiments:
    - instance => string representing problem used
    - config:
       - name: can be empty, but useful to have a simple name for the config
       - solver: e.g. class name
       - params: hyperparameters used (nested dict whose leaves should be hashable, and preferably jsonable)
    - status: status of tge solver at then of the solve process
    - metrics: timeseries of objective, bound, ...


    """

    @abstractmethod
    def get_new_experiment_id(self) -> int:
        ...

    @abstractmethod
    def store(self, xp: Experiment) -> None:
        """Store the experiment in the database.

        Could store a complete experiment.
        Depending on implementations, could also support storing a partial experiment,
        and then overwriting with a complete experiment by re-calling `store` on an experiment
        with same id and more data.
        """
        ...

    @abstractmethod
    def load(self) -> list[Experiment]:
        """Load all experiments of the study."""
        ...

    def load_results(self) -> list[pd.DataFrame]:
        """Load all experiments as time-indexes dataframes with metadata in `attrs` attribute."""
        return [xp.to_df() for xp in self.load()]

    def close(self) -> None:
        """Close the database."""
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()


# database keys
METADATA = "metadata"


COL_STR_MAXSIZE = {
    REASON: 512,
    INSTANCE: 32,
    STATUS: max(len(s.value) for s in StatusSolver),
    CONFIG_PREFIX + SOLVER: 52,
    CONFIG_PREFIX + NAME: 32,
    CONFIG_PREFIX + PARAMETERS: 1024,
}

PARAMETERS_STR_MAX_SIZE = 1024
REASON_MAX_SIZE = 512


def _get_metrics_key(xp_id: int) -> str:
    return f"{METRICS}/{xp_id}"


class Hdf5Database(Database):
    """Database based on hdf5 format."""

    def __init__(self, filepath: str):
        self.hdfstore = pd.HDFStore(filepath)
        if ID not in self.hdfstore:
            self.hdfstore[ID] = pd.Series([-1])

    def close(self) -> None:
        self.hdfstore.close()

    def get_new_experiment_id(self) -> int:
        current_id = int(self.hdfstore[ID].iloc[-1])
        new_id = current_id + 1
        self.hdfstore[ID] = pd.Series([new_id])
        return new_id

    def store(self, xp: Experiment) -> None:
        self.store_metadata(xp)
        self.store_metrics(xp)

    def store_metadata(self, xp: Experiment) -> None:
        self.hdfstore.append(
            METADATA,
            pd.DataFrame.from_records([xp.get_metadata_as_a_record()]),
            min_itemsize=COL_STR_MAXSIZE,
        )

    def store_metrics(self, xp: Experiment) -> None:
        self.hdfstore.put(_get_metrics_key(xp.xp_id), xp.metrics)

    def load(self) -> list[Experiment]:
        df_metadata: pd.DataFrame = self.hdfstore[METADATA]
        xps: list[Experiment] = []
        for row in df_metadata.itertuples(index=False):
            record = row._asdict()
            xp_id = record[ID]
            config = SolverJsonableConfig.from_xp_metadata_record(record)
            try:
                metrics = self.hdfstore[_get_metrics_key(xp_id)]
            except KeyError:
                metrics = pd.DataFrame()
                logger.warning(
                    f"Missing metrics for xp {xp_id}: config {config.name}, instance {record[INSTANCE]}"
                )
            xps.append(
                Experiment(
                    xp_id=xp_id,
                    instance=record[INSTANCE],
                    status=record[STATUS],
                    metrics=metrics,
                    config=config,
                    reason=record[REASON],
                )
            )
        return xps
