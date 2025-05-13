from collections import defaultdict
from collections.abc import Container, Iterable
from copy import copy
from typing import Any, Optional, Union

import pandas as pd

from discrete_optimization.generic_tools.dashboard.config import ConfigStore
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study.experiment import (
    CONFIG,
    INSTANCE,
    REASON,
    STATUS,
)

# data columns
BOUND = "bound"
OBJ = "obj"
GAP = "gap"
GAP_REL_OBJ = "gap_rel_obj"
GAP_REL_BOUND = "gap_rel_bound"
GAP_REL_MAX_OBJ_BOUND = "gap_rel_max_obj_bound"
FIT = "fit"

# available stats
MEAN = "mean"
MAX = "max"
MIN = "min"
MEDIAN = "median"
QUANTILE = "quantile"
map_stat_key2func = {
    MEAN: pd.DataFrame.mean,
    MAX: pd.DataFrame.max,
    MIN: pd.DataFrame.min,
    MEDIAN: pd.DataFrame.median,
    QUANTILE: pd.DataFrame.quantile,
}

# new metadata keys for empty xps
I_RUN_LABEL = "attempt"

TIMEOUT_REASON = "Probably due to a timeout."


def drop_empty_results(results: list[pd.DataFrame]) -> list[pd.DataFrame]:
    return [df for df in results if len(df) > 0]


def extract_empty_xps_metadata(results: list[pd.DataFrame]) -> pd.DataFrame:
    empty_xps_attrs = [df.attrs for df in results if len(df) == 0]
    metadata_df = pd.DataFrame(
        empty_xps_attrs, columns=[CONFIG, INSTANCE, I_RUN_LABEL, STATUS, REASON]
    )
    metadata_df.loc[metadata_df.loc[:, REASON] == "", REASON] = TIMEOUT_REASON
    metadata_df.loc[:, STATUS] = metadata_df.loc[:, STATUS].map(lambda st: st.value)
    return metadata_df


def normalize_results(results: list[pd.DataFrame], config_store: ConfigStore) -> None:
    # add all configs beforehand to ensure name<->config bijection
    for df in results:
        config = df.attrs[CONFIG]
        if isinstance(config, dict):
            config_store.add(config)
        elif not isinstance(config, str):
            raise ValueError(
                "For each result df, df.attrs['config'] must be either a dictionary "
                "or a string representing its name."
            )

    # normalize each result one by one
    for df in results:
        normalize_df(df=df, config_store=config_store)

    # compute an attempt number
    n_runs_by_config_instance = defaultdict(lambda: 0)
    for df in results:
        attrs = df.attrs
        key = attrs[CONFIG], attrs[INSTANCE]
        attrs[I_RUN_LABEL] = n_runs_by_config_instance[key]
        n_runs_by_config_instance[key] += 1


def normalize_df(
    df: pd.DataFrame, config_store: ConfigStore, timedelta_unit="s"
) -> None:
    normalize_metadata(df.attrs, config_store=config_store)
    df.sort_index(inplace=True)
    # -> timedeltaindex
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index - df.index[0]
    elif not isinstance(df.index, pd.TimedeltaIndex):
        df.index = pd.to_timedelta(df.index, unit=timedelta_unit)
    # -> total_seconds
    df.index = [t.total_seconds() for t in df.index.to_pytimedelta()]
    df.index.name = "time (s)"
    # drop columns full of NaNs
    df.dropna(axis=1, how="all", inplace=True)
    # no more columns => on vide la dataframe
    if len(df.columns) == 0:
        df.drop(df.index, inplace=True)


def normalize_metadata(metadata: dict[str, Any], config_store: ConfigStore) -> None:
    # status
    if STATUS not in metadata:
        metadata[STATUS] = StatusSolver.UNKNOWN
    else:
        if not isinstance(metadata[STATUS], StatusSolver):
            metadata[STATUS] = StatusSolver(metadata[STATUS].upper())
    # config -> config name
    config = metadata[CONFIG]
    if isinstance(config, dict):
        config_name = config_store.get_name(config)
    elif isinstance(config, str):
        config_name = config
    else:
        raise ValueError(
            "For each result df, df.attrs['config'] must be either a dictionary "
            "or a string representing its name."
        )
    metadata[CONFIG] = config_name


def compute_extra_metrics(results: list[pd.DataFrame]) -> None:
    for df in results:
        compute_extra_metrics_df(df)


def compute_extra_metrics_df(df: pd.DataFrame) -> None:
    # gap
    if GAP not in df.columns:
        if OBJ in df.columns and BOUND in df.columns:
            df[GAP] = (df[OBJ] - df[BOUND]).abs()

    # gap relative
    if GAP in df.columns:
        if GAP_REL_OBJ not in df.columns:
            if OBJ in df.columns:
                df[GAP_REL_OBJ] = df[GAP] / df[OBJ].abs()

        if GAP_REL_BOUND not in df.columns:
            if BOUND in df.columns:
                df[GAP_REL_BOUND] = df[GAP] / df[BOUND].abs()

        if GAP_REL_OBJ not in df.columns:
            if OBJ in df.columns and BOUND in df.columns:
                df[GAP_REL_MAX_OBJ_BOUND] = df[GAP] / df.loc[:, [OBJ, BOUND]].abs().max(
                    axis=1
                )


def extract_instances(results) -> set[str]:
    return {df.attrs[INSTANCE] for df in results}


def extract_configs(results) -> set[str]:
    return {df.attrs[CONFIG] for df in results}


def extract_metrics(results) -> set[str]:
    metrics = set()
    for df in results:
        metrics.update(df.columns)
    return metrics


def filter_results(
    results: list[pd.DataFrame], configs: Container[str], instances: Container[str]
) -> list[pd.DataFrame]:
    return [
        df
        for df in results
        if df.attrs[INSTANCE] in instances and df.attrs[CONFIG] in configs
    ]


def clip_df(df: pd.DataFrame, clip_value: float) -> pd.DataFrame:
    # copy
    df = copy(df)
    df[df > clip_value] = pd.NA
    df[df < -clip_value] = pd.NA
    return df


def clip_results(results: list[pd.DataFrame], clip_value: float) -> list[pd.DataFrame]:
    return [clip_df(df, clip_value=clip_value) for df in results]


def get_experiment_name(df: Union[pd.DataFrame, pd.Series], with_run_nb=True) -> str:
    if with_run_nb:
        return f"{df.attrs[CONFIG]} x {df.attrs[INSTANCE]} #{df.attrs[I_RUN_LABEL]}"
    else:
        return f"{df.attrs[CONFIG]} x {df.attrs[INSTANCE]}"


def has_multiple_runs(results: list[pd.DataFrame]) -> bool:
    return max(df.attrs[I_RUN_LABEL] for df in results) > 0


def get_stat_name(stat: str, q: float) -> str:
    if stat == QUANTILE:
        return f"{stat}({q})"
    else:
        return stat


def aggregate_results_config(results: list[pd.DataFrame], config: str) -> pd.DataFrame:
    results_config = [df for df in results if df.attrs[CONFIG] == config]
    instances_config = [
        (df.attrs[INSTANCE], run) for run, df in enumerate(results_config)
    ]
    return (
        pd.concat(
            results_config,
            axis=1,
            join="outer",
            keys=instances_config,
            names=["instance", "run"],
        )
        .sort_index()  # increasing time
        .ffill()  # replace nan by previous value (as instances do not share same timescale)
        .stack(level=-1, future_stack=True)  # pre-stack to speed-up stat computation
    )


def aggregate_results_by_config(
    results: list[pd.DataFrame], configs: Iterable[str]
) -> dict[str, pd.DataFrame]:
    return {
        config: aggregate_results_config(results=results, config=config)
        for config in configs
    }


def compute_stat_by_config(
    results_by_config: dict[str, pd.DataFrame],
    stat: str = MEAN,
    q: float = 0.5,
    instances: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    return {
        config: compute_stat_from_df_config(
            df_config, stat=stat, q=q, instances=instances
        )
        for config, df_config in results_by_config.items()
        if len(df_config) > 0
    }


def compute_stat_from_df_config(
    df_config: pd.DataFrame,
    stat: str = MEAN,
    q: float = 0.5,
    instances: Optional[list[str]] = None,
) -> pd.DataFrame:
    # filter instances to aggregate
    if instances is not None:
        # intersection with instances present in df_config (some could be missing for a given config)
        df_config_instances = set(df_config.columns.get_level_values("instance"))
        instances = [i for i in instances if i in df_config_instances]
        # filter data according to instances
        df_config = df_config.loc[:, instances]
        # no remaining instances? => empty result
        if len(df_config.columns) == 0:
            empty_df = pd.DataFrame([])
            empty_df.index.name = df_config.index.names[0]
            empty_df.columns.name = df_config.index.names[1]
            return empty_df
    # function used to aggregate
    stat_func = map_stat_key2func[stat]
    if stat_func is pd.DataFrame.quantile:
        kwargs = dict(q=q)
    else:
        kwargs = dict()
    # aggregate
    df_stat = stat_func(df_config.dropna(), axis=1, **kwargs).unstack()
    return df_stat


def extract_solvetimes_by_config(results: list[pd.DataFrame]) -> dict[str, list[float]]:
    solvetimes_by_config = defaultdict(list)
    for df in results:
        if df.attrs[STATUS] == StatusSolver.OPTIMAL:
            solvetime = df.index[-1]
            config = df.attrs[CONFIG]
            solvetimes_by_config[config].append(solvetime)
    return solvetimes_by_config


def convert_solvetimes2nbsolvedinstances(
    solvetimes: list[float],
    time_label: str = "time",
    name: str = "nb of solved instances",
) -> pd.Series:
    ser = pd.Series(
        {t: n for t, n in zip(sorted(solvetimes), range(1, len(solvetimes) + 1))},
        name=name,
    )
    ser.index.name = time_label
    return ser


def extract_nbsolvedinstances_by_config(
    results: list[pd.DataFrame],
) -> dict[str, pd.Series]:
    if len(results) == 0:
        time_label = "time"
    else:
        time_label = results[0].index.name
    return {
        config: convert_solvetimes2nbsolvedinstances(solvetimes, time_label=time_label)
        for config, solvetimes in extract_solvetimes_by_config(results=results).items()
    }
