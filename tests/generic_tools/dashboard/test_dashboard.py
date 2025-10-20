import logging
import os
from copy import copy
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
    instances = [
        "gc_50_3",
        "gc_50_1",
        "gc_50_3",
    ]  # test duplicates
    cpsat_integer_timeout = False
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
    with (
        Hdf5Database(database_filepath) as database
    ):  # ensure closing the database at the end of computation (even if error)
        # loop over instances x configs
        for instance in instances:
            for config_name, solver_config in solver_configs.items():
                logging.info(
                    f"###### Instance {instance}, config {config_name} ######\n\n"
                )
                # remove one xp to have solvers with less instances
                if config_name == "mathopt" and instance == "gc_50_1":
                    continue

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

                # simulate a timeout for one instance of cpsat-integer
                if config_name == "cpsat-integer" and not cpsat_integer_timeout:
                    cpsat_integer_timeout = True
                    metrics = pd.DataFrame([])
                    status = StatusSolver.UNKNOWN

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
                # modify fake config to create a duplicate
                if config_name == "fake" and instance == "gc_50_1":
                    xp.config.parameters = copy(xp.config.parameters)
                    xp.config.parameters["toto"] = 0.5
                # modify fitness to have different ranks
                elif config_name == "mathopt":
                    xp.metrics["fit"] -= 1  # degrade fitness
                    xp.status = StatusSolver.SATISFIED.value  # not optimal anymore
                elif config_name == "cpsat-binary" and instance == "gc_50_1":
                    xp.metrics["fit"] += 1  # improve fitness
                elif instance == "gc_50_1" and xp.status == StatusSolver.OPTIMAL.value:
                    xp.status = StatusSolver.SATISFIED.value  # not optimal anymore
                elif config_name == "cpsat-binary" and instance == "gc_50_3":
                    xp.status = (
                        StatusSolver.SATISFIED.value
                    )  # make it unaware of optimality
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
    assert len(app.results) == 7
    assert len(app.full_results) == 14


def test_update_config_display_ok(app):
    string = app.update_config_display(config_name="cpsat-binary")
    assert '"solver": "CpSatColoringSolver"' in string
    assert '"parameters_cp": {' in string
    assert '"multiprocess": true,' in string


def test_update_config_display_no_config(app):
    string = app.update_config_display(config_name="toto-binary")
    assert "WARNING: no config" in string


def test_update_config_display_several_configs(app):
    string = app.update_config_display(config_name="fake")
    assert "WARNING: 2 configs" in string


def test_replace_instances_aliases(app):
    configs = app.full_configs
    instances = ["@all"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["@withsol"]
    assert app._replace_instances_aliases(instances, configs) == []
    instances = ["@withsol", "gc_50_3", "gc_50_1"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_3", "gc_50_1"]

    configs = app.configs
    instances = ["@all"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["@withsol"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_3"]
    instances = ["@withsol", "gc_50_1", "gc_50_3"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["gc_50_1", "@withsol"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["@withsol", "gc_50_3"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_3"]

    configs = ["cpsat-binary"]
    instances = ["@all"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["@withsol"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_1", "gc_50_3"]
    instances = ["@withsol", "gc_50_3", "gc_50_1"]
    assert app._replace_instances_aliases(instances, configs) == ["gc_50_3", "gc_50_1"]


@pytest.mark.parametrize(
    "configs, instances, metric, time_log_scale, metric_log_scale, expected_n_traces, attempt_in_legend",
    [
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            False,
            False,
            7,
            True,
        ),
        (["cpsat-binary"], ["gc_50_1"], "gap", False, False, 1, False),
        (
            ["cpsat-binary", "cpsat-integer", "mathopt"],
            ["@withsol"],
            "fit",
            True,
            True,
            5,
            True,
        ),
    ],
)
def test_update_graph_metric(
    app,
    configs,
    instances,
    metric,
    time_log_scale,
    metric_log_scale,
    expected_n_traces,
    attempt_in_legend,
):
    plot = app.update_graph_metric(
        configs=configs,
        instances=instances,
        metric=metric,
        time_log_scale=time_log_scale,
        metric_log_scale=metric_log_scale,
        clip_value=1e50,
    )
    assert len(plot.data) == expected_n_traces
    assert ("#" in plot.data[0].name) is attempt_in_legend


@pytest.mark.parametrize(
    "configs, instances, metric, stat, time_log_scale, metric_log_scale, min_xp_proportion, expected_n_traces, nodata",
    [
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            "mean",
            False,
            False,
            1.0,
            3,
            False,
        ),
        (["cpsat-binary"], ["gc_50_3"], "gap", "quantile", False, False, 1.0, 1, False),
        (["cpsat-binary"], [], "gap", "quantile", True, True, 1.0, 0, True),
        (["cpsat-binary"], [], "gap", "quantile", True, True, 0.5, 0, True),
    ],
)
def test_update_graph_agg_metric(
    app,
    configs,
    instances,
    metric,
    stat,
    time_log_scale,
    metric_log_scale,
    min_xp_proportion,
    expected_n_traces,
    nodata,
):
    output = app.update_graph_agg_metric(
        configs=configs,
        instances=instances,
        metric=metric,
        stat=stat,
        q=0.5,
        time_log_scale=time_log_scale,
        metric_log_scale=metric_log_scale,
        min_xp_proportion=min_xp_proportion,
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
            assert pd.isna(df[df.config == "timeout"].iloc[0, 3:]).all()
            assert (df[df.config == "timeout"]["# xps"] == 3).all()
            assert (df[df.config == "timeout"]["# no sol"] == 3).all()
        if "cpsat-integer" in configs:
            assert (df[df.config == "cpsat-integer"]["# xps"] == 3).all()
            assert (df[df.config == "cpsat-integer"]["# no sol"] == 1).all()
        if "cpsat-binary" in configs:
            assert (df[df.config == "cpsat-binary"]["# no sol"] == 0).all()


@pytest.mark.parametrize(
    "configs, instances, metric, stat, minimizing, dist_label, all_xps",
    [
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            "mean",
            False,
            "dist rel to best",
            True,
        ),
        (
            ["cpsat-binary", "cpsat-integer", "fake", "mathopt", "timeout"],
            ["@all"],
            "fit",
            "quantile",
            True,
            "dist to best",
            True,
        ),
        (
            ["cpsat-binary"],
            ["gc_50_3"],
            "gap",
            "quantile",
            True,
            "dist rel to best",
            False,
        ),
    ],
)
def test_update_table_rank_agg(
    app, configs, instances, metric, stat, minimizing, dist_label, all_xps
):
    output = app.update_table_rank_agg(
        configs=configs,
        instances=instances,
        metric=metric,
        stat=stat,
        q=0.5,
        minimizing=minimizing,
        clip_value=1e50,
        transpose=False,
        time_log_scale=False,
        time_label="convergence time (s)",
        dist_label=dist_label,
        all_xps=all_xps,
    )
    df = pd.DataFrame(output["data"])
    if "@all" in instances:
        nb_instances = 2
    else:
        nb_instances = len(instances)
    if metric == "fit":
        if "mathopt" in configs:
            if minimizing:
                assert (df[df.config == "mathopt"]["dist to best"] == 0).all()
                assert (df[df.config == "mathopt"]["dist rel to best"] == 0).all()
            else:
                assert (df[df.config == "mathopt"]["dist to best"] == 1).all()
                assert (df[df.config == "mathopt"]["dist rel to best"] == 1 / 6).all()
                assert (df[df.config == "mathopt"]["# 3"] == 1).all()
        if "cpsat-integer" in configs:
            if minimizing:
                assert (
                    df[df.config == "cpsat-integer"]["dist rel to best"] == 1 / 14
                ).all()
            else:
                assert (
                    df[df.config == "cpsat-integer"]["dist rel to best"] == 1 / 6
                ).all()
            assert (df[df.config == "cpsat-integer"]["dist to best"] == 0.5).all()
            assert (df[df.config == "cpsat-integer"]["# 1"] == 1).all()
        if minimizing:
            assert (df[df.config == "cpsat-binary"]["dist to best"] == 1).all()
            assert (df[df.config == "cpsat-binary"]["dist rel to best"] == 1 / 7).all()
            assert (df[df.config == "cpsat-binary"]["# 1"] == 0).all()
        else:
            assert (df[df.config == "cpsat-binary"]["dist to best"] == 0).all()
            assert (df[df.config == "cpsat-binary"]["dist rel to best"] == 0).all()
            assert (df[df.config == "cpsat-binary"]["# 1"] == nb_instances).all()
    elif metric == "gap":
        assert (df[df.config == "cpsat-binary"]["# 1"] == nb_instances).all()
        assert (df[df.config == "cpsat-binary"]["dist to best"] == 0).all()

    if "fake" in configs:
        assert (df[df.config == "fake"]["# failed"] == nb_instances).all()

    assert "convergence time (s)" in df


@pytest.mark.parametrize(
    "time_log_scale",
    [True, False],
)
@pytest.mark.parametrize(
    "transpose",
    [True, False],
)
@pytest.mark.parametrize(
    "include_solved_wo_proof",
    [False, True],
)
def test_update_graph_nb_solved_instances(
    app, time_log_scale, transpose, include_solved_wo_proof
):
    configs = ["mathopt", "cpsat-binary", "cpsat-integer", "timeout"]
    expected_n_traces = 2
    output = app.update_graph_nb_solved_instances(
        configs=configs,
        time_log_scale=time_log_scale,
        transpose=transpose,
        include_solved_wo_proof=include_solved_wo_proof,
    )
    plot = output["plot"]
    assert len(plot.data) == expected_n_traces
    if transpose:
        assert plot.layout.xaxis.title.text == "% of solved instances"
        assert plot.layout.yaxis.title.text == "time (s)"
        time_axes = next(plot.select_yaxes())
    else:
        assert plot.layout.yaxis.title.text == "% of solved instances"
        assert plot.layout.xaxis.title.text == "time (s)"
        time_axes = next(plot.select_xaxes())

    if time_log_scale:
        assert time_axes.type == "log"
    else:
        assert time_axes.type is None

    df = pd.DataFrame(output["data"])
    assert [c["name"] for c in output["columns"]] == df.columns.tolist()
    assert (df[df.config == "cpsat-integer"].iloc[0, 1:].values == [1, 3]).all()
    if include_solved_wo_proof:
        assert (df[df.config == "cpsat-binary"].iloc[0, 1:].values == [3, 3]).all()
    else:
        assert (df[df.config == "cpsat-binary"].iloc[0, 1:].values == [1, 3]).all()
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
        ("cpsat-binary", "gc_50_3", 1, False, StatusSolver.SATISFIED),
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
    assert len(df) == 7
    assert sum(df["status"] == "FAILED") == 3
    assert sum(df["status"] == "UNKNOWN") == 4
    assert "timeout" in df.loc[df["status"] == "UNKNOWN", "reason"].iloc[0]
    assert "fake solver" in df.loc[df["status"] == "FAILED", "reason"].iloc[0]
    assert "RuntimeError" in df.loc[df["status"] == "FAILED", "reason"].iloc[0]
