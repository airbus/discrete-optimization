import logging
import os
from typing import Any, Optional

import pandas as pd
import pytest
from pytest import fixture

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.coloring.solvers.lp import MathOptColoringSolver
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.dashboard.dashboard import (
    TIME_LOGSCALE_KEY,
    TRANSPOSE_KEY,
)
from discrete_optimization.generic_tools.do_solver import (
    BoundsProviderMixin,
    StatusSolver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)


class FakeFailingSolver(ColoringSolver):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        raise RuntimeError("This fake solver is failing!")


class FakeTimeoutSolver(BoundsProviderMixin, ColoringSolver):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        res = self.create_result_storage()  # empty sols
        ...  # timeout

        callbacks_list.on_solve_end(solver=self, res=res)
        return res

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return None

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return None


def run_study(study_name):
    instances = ["gc_50_3", "gc_50_1", "gc_50_3"]  # test duplicates
    solver_configs = {
        "cpsat-integer": SolverConfig(
            cls=CpSatColoringSolver,
            kwargs=dict(
                parameters_cp=ParametersCp.default_cpsat(),
                modeling=ModelingCpSat.INTEGER,
                do_warmstart=False,
                value_sequence_chain=False,
                used_variable=True,
                symmetry_on_used=True,
            ),
        ),
        "cpsat-binary": SolverConfig(
            cls=CpSatColoringSolver,
            kwargs=dict(
                parameters_cp=ParametersCp.default_cpsat(),
                modeling=ModelingCpSat.BINARY,
                do_warmstart=False,
                value_sequence_chain=False,
                used_variable=True,
                symmetry_on_used=True,
            ),
        ),
        "fake": SolverConfig(cls=FakeFailingSolver, kwargs=dict()),
        "mathopt": SolverConfig(cls=MathOptColoringSolver, kwargs=dict()),
        "timeout": SolverConfig(cls=FakeTimeoutSolver, kwargs=dict()),
    }
    database_filepath = f"{study_name}.h5"
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass
    with Hdf5Database(
        database_filepath
    ) as database:  # ensure closing the database at the end of computation (even if error)

        # loop over instances x configs
        for instance in instances:
            for config_name, solver_config in solver_configs.items():

                logging.info(
                    f"###### Instance {instance}, config {config_name} ######\n\n"
                )

                try:
                    # init problem
                    file = [f for f in get_data_available() if instance in f][0]
                    color_problem = parse_file(file)
                    # init solver
                    stats_cb = StatsWithBoundsCallback()
                    solver = solver_config.cls(color_problem, **solver_config.kwargs)
                    solver.init_model(**solver_config.kwargs)
                    # solve
                    result_store = solver.solve(
                        callbacks=[
                            stats_cb,
                            NbIterationTracker(step_verbosity_level=logging.INFO),
                        ],
                        **solver_config.kwargs,
                    )
                except Exception as e:
                    # failed experiment
                    metrics = pd.DataFrame([])
                    status = StatusSolver.ERROR
                    reason = f"{type(e).__name__}: {str(e)}"
                else:
                    # get metrics and solver status
                    status = solver.status_solver
                    metrics = stats_cb.get_df_metrics()
                    reason = ""

                # store corresponding experiment
                xp_id = database.get_new_experiment_id()
                xp = Experiment.from_solver_config(
                    xp_id=xp_id,
                    instance=instance,
                    config_name=config_name,
                    solver_config=solver_config,
                    metrics=metrics,
                    status=status,
                    reason=reason,
                )
                database.store(xp)


@fixture(scope="module")
def study_name():
    # create and run experiments, and fill a database with it
    study_name = "Coloring-Test-Study"
    run_study(study_name=study_name)
    return study_name


@fixture
def study_results(study_name):
    # retrieve data
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()

    return results


@fixture
def app(study_results):
    return Dashboard(results=study_results)


def test_init_app(app):
    assert len(app.results) == 9
    assert len(app.full_results) == 15


def test_update_config_display_ok(app):
    string = app.update_config_display(config_name="cpsat-binary")
    assert '"solver": "CpSatColoringSolver"' in string
    assert '"parameters_cp": {' in string
    assert '"multiprocess": true,' in string


def test_update_config_display_nok(app):
    string = app.update_config_display(config_name="toto-binary")
    assert "NOT_FOUND" in string


@pytest.mark.parametrize(
    "configs, instances, metric, time_log_scale, expected_n_traces, attempt_in_legend",
    [
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            [],
            9,
            True,
        ),
        (["cpsat-binary"], ["gc_50_1"], "gap", [], 1, False),
    ],
)
def test_update_graph_metric(
    app,
    configs,
    instances,
    metric,
    time_log_scale,
    expected_n_traces,
    attempt_in_legend,
):
    plot = app.update_graph_metric(
        configs=configs,
        instances=instances,
        metric=metric,
        time_log_scale=time_log_scale,
        clip_value=1e50,
    )
    assert len(plot.data) == expected_n_traces
    assert ("#" in plot.data[0].name) is attempt_in_legend


@pytest.mark.parametrize(
    "configs, instances, metric, stat, time_log_scale, expected_n_traces, nodata",
    [
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            "mean",
            [],
            3,
            False,
        ),
        (["cpsat-binary"], ["gc_50_3"], "gap", "quantile", [], 1, False),
        (["cpsat-binary"], [], "gap", "quantile", [], 0, True),
    ],
)
def test_update_graph_agg_metric(
    app, configs, instances, metric, stat, time_log_scale, expected_n_traces, nodata
):
    output = app.update_graph_agg_metric(
        configs=configs,
        instances=instances,
        metric=metric,
        stat=stat,
        q=0.5,
        time_log_scale=time_log_scale,
        clip_value=1e50,
    )
    plot = output["plot"]
    assert len(plot.data) == expected_n_traces
    if nodata:
        assert len(plot.layout.annotations) == 1
        assert plot.layout.annotations[0].text == "NO DATA"
    else:
        assert len(plot.layout.annotations) == 0

    df = pd.DataFrame(output["data"])
    assert [c["name"] for c in output["columns"]] == df.columns.tolist()
    assert all(config in df.config.values for config in configs)
    if len(instances) > 0:
        assert df.loc[df.config == "cpsat-binary", "gap"].values == 0
        if "timeout" in configs:
            assert pd.isna(df[df.config == "timeout"].iloc[0, 1:]).all()


@pytest.mark.parametrize(
    "time_log_scale",
    [[TIME_LOGSCALE_KEY], []],
)
@pytest.mark.parametrize(
    "transpose_value",
    [[TRANSPOSE_KEY], []],
)
def test_update_graph_nb_solved_instances(app, time_log_scale, transpose_value):
    configs = ["mathopt", "cpsat-integer", "timeout"]
    expected_n_traces = 2
    output = app.update_graph_nb_solved_instances(
        configs=configs, time_log_scale=time_log_scale, transpose_value=transpose_value
    )
    plot = output["plot"]
    assert len(plot.data) == expected_n_traces
    if TRANSPOSE_KEY in transpose_value:
        assert plot.layout.xaxis.title.text == "% of solved instances"
        assert plot.layout.yaxis.title.text == "time (s)"
        time_axes = next(plot.select_yaxes())
    else:
        assert plot.layout.yaxis.title.text == "% of solved instances"
        assert plot.layout.xaxis.title.text == "time (s)"
        time_axes = next(plot.select_xaxes())

    if TIME_LOGSCALE_KEY in time_log_scale:
        assert time_axes.type == "log"
    else:
        assert time_axes.type is None

    df = pd.DataFrame(output["data"])
    assert [c["name"] for c in output["columns"]] == df.columns.tolist()
    assert (df[df.config == "cpsat-integer"].iloc[0, 1:].values == [3, 3]).all()
    assert (df[df.config == "timeout"].iloc[0, 1:].values == [0, 3]).all()


@pytest.mark.parametrize(
    "instance, nb_runs",
    [
        (
            "gc_50_1",
            1,
        ),
        (
            "gc_50_3",
            2,
        ),
    ],
)
def test_update_run_options(app, instance, nb_runs):
    config = "cpsat-binary"
    output = app.update_run_options(
        config=config,
        instance=instance,
    )
    assert len(output["options"]) == nb_runs


@pytest.mark.parametrize(
    "config, instance, i_run, nodata, status",
    [
        ("cpsat-binary", "gc_50_1", 0, False, StatusSolver.OPTIMAL),
        ("cpsat-binary", "gc_50_3", 1, False, StatusSolver.OPTIMAL),
        ("fake", "gc_50_1", 0, True, StatusSolver.ERROR),
        ("timeout", "gc_50_1", 0, True, StatusSolver.UNKNOWN),
    ],
)
def test_update_xp_data(app, config, instance, i_run, nodata, status):
    output = app.update_xp_data(config=config, instance=instance, run=i_run)
    if nodata:
        assert len(output["data"]) == 0
        assert output["nodata"] == "d-block"
    else:
        assert len(output["data"]) > 0
        assert output["nodata"] == "d-none"
    assert status.value in output["status"]


def test_empty_xps(app):
    df = app.empty_xps_metadata
    assert len(df) == 6
    assert sum(df["status"] == "FAILED") == 3
    assert sum(df["status"] == "UNKNOWN") == 3
    assert "timeout" in df.loc[df["status"] == "UNKNOWN", "reason"].iloc[0]
    assert "fake solver" in df.loc[df["status"] == "FAILED", "reason"].iloc[0]
    assert "RuntimeError" in df.loc[df["status"] == "FAILED", "reason"].iloc[0]
