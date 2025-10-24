from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Optional, Union

import pandas as pd

from discrete_optimization.generic_tools.dashboard.config import ConfigStore
from discrete_optimization.generic_tools.dashboard.plots import (
    create_graph_from_series_dict,
    create_solvers_competition_graph,
)
from discrete_optimization.generic_tools.dashboard.preprocess import (
    CONFIG,
    CONVERGENCE_TIME,
    DIST_TO_BEST,
    DIST_TO_BEST_REL,
    FIT,
    MEAN,
    QUANTILE,
    aggregate_results_by_config,
    clip_df,
    clip_results,
    compute_best_metrics_by_xp,
    compute_convergence_time_by_xp,
    compute_extra_metrics,
    compute_stat_by_config,
    compute_summary_agg_ranks_and_dist_to_best_metric,
    construct_summary_metric_agg,
    construct_summary_nbsolved_instances,
    convert_nb2percentage_solvedinstances_by_config,
    drop_empty_results,
    extract_configs,
    extract_empty_xps_metadata,
    extract_instances,
    extract_instances_with_sol_by_config,
    extract_metrics,
    extract_nb_xps_by_config,
    extract_nb_xps_w_n_wo_sol_by_config,
    extract_nbsolvedinstances_by_config,
    extract_nbsolvedinstances_wo_status_by_config,
    filter_results,
    get_experiment_name,
    get_status_str,
    has_multiple_runs,
    map_stat_key2func_df,
    normalize_results,
)

try:
    import dash
except ImportError:
    dash_available = False
    Dash = object
else:
    dash_available = True
    from dash import Dash, Input, Output, callback, ctx, dash_table, dcc, html

try:
    import dash_bootstrap_components as dbc
except ImportError:
    dbc_available = False
else:
    dbc_available = True


TIME_LOGSCALE_KEY = "time log-scale"
TIME_LOGSCALE_ID = "time-log-scale"
TIME_LOGSCALE_DIV_ID = "time-log-scale-div"

MINIMIZING_KEY = "minimizing"
MINIMIZING_ID = "minimizing"

SOLVED_WO_PROOF_ID = "solved-wo-proof"
SOLVED_WO_PROOF_KEY = "w/o proof"
SOLVED_WO_PROOF_DIV_ID = "solved-wo-proof-div"

TRANSPOSE_KEY = "transpose"
TRANSPOSE_ID = "transpose"

SOLVERS_COMPETITION_ALL_XPS_KEY = "all xps"
SOLVERS_COMPETITION_ALL_XPS_ID = "solvers_competition-all-xps"
SOLVERS_COMPETITION_DIST_ID = "solvers-competition-dist"
SOLVERS_COMPETITION_TIME_ID = "solvers-competition-time"
SOLVERS_COMPETITION_ALL_XPS_DIV_ID = "solvers_competition-all-xps-div"
SOLVERS_COMPETITION_DIST_DIV_ID = "solvers-competition-dist-div"
SOLVERS_COMPETITION_TIME_DIV_ID = "solvers-competition-time-div"

GRAPH_METRIC_ID = "graph-metric"
GRAPH_AGG_METRIC_ID = "graph-agg-metric"
GRAPH_NB_SOLVED_INSTANCES_ID = "graph-nb-solved-instances"
GRAPH_SOLVERS_COMPETITION_ID = "graph-solvers-competition"

TABLE_AGG_METRIC_ID = "table-agg-metric"
TABLE_AGG_RANK_ID = "table-agg-rank"
TABLE_EMPTY_XPS_ID = "table-empty-xps"
TABLE_XP_ID = "table-metric"
TABLE_XP_NO_DATA_ID = "table-metric-no-data"
TABLE_XP_STATUS_ID = "table-metric-status"
TABLE_NB_SOLVED_INSTANCES_ID = "table-nb-solved-instances"
CONFIG_MD_ID = "config-markdown"

TAB_METRIC_ID = "tab-metric"
TAB_AGG_METRIC_ID = "tab-agg-metric"
TAB_NB_SOLVED_INSTANCES_ID = "tab-nb-solved-instances"
TAB_CONFIG_ID = "tab-config"
TAB_XP_ID = "tab-experiment"
TAB_EMPTY_XPS_ID = "tab-empty-xps"
TAB_AGG_RANK_ID = "tab-agg-rank"
TABS_ID = "tabs"

ALIAS_INSTANCES_ALL = "@all"
ALIAS_INSTANCES_WITHSOL = "@withsol"
ALIASES_INSTANCES = [ALIAS_INSTANCES_ALL, ALIAS_INSTANCES_WITHSOL]

METRIC_ID = "metric"
INSTANCES_ID = "instances"
STAT_ID = "stat"
CONFIGS_ID = "configs"
CONFIG_ID = "config"
Q_ID = "quantile-q"
CLIP_ID = "clip"
CLIP_DIV_ID = "clip-div"
METRIC_DIV_ID = "metric-div"
INSTANCES_DIV_ID = "instances-div"
STAT_DIV_ID = "stat-div"
CONFIG_DIV_ID = "config-div"
CONFIGS_DIV_ID = "configs-div"
Q_DIV_ID = "quantile-q-div"
TRANSPOSE_DIV_ID = "transpose-div"
MINIMIZING_DIV_ID = "minimizing-div"
INSTANCE_ID = "instance"
INSTANCE_DIV_ID = "instance-div"
RUN_ID = "run-id"
RUN_DIV_ID = "run-id-div"
MIN_XP_PROP_ID = "min-exp-proportion-slider"
MIN_XP_PROP_DIV_ID = "min-exp-proportion-div"

Y_LOGSCALE_KEY = "metric log-scale"
Y_LOGSCALE_ID = "y-log-scale"
Y_LOGSCALE_DIV_ID = "y-log-scale-div"


class Dashboard(Dash):
    def __init__(
        self,
        results: Optional[list[pd.DataFrame]] = None,
        title="Discrete-Optimization Experiments Dashboard",
        external_stylesheets: Optional[Sequence[Union[str, dict[str, Any]]]] = None,
        **kwargs,
    ):
        if not dash_available:
            raise RuntimeError("You need to install 'dash' to create a dashboard.")
        if not dbc_available:
            raise RuntimeError(
                "You need to install 'dash_bootstrap_components' to create a dashboard."
            )

        if external_stylesheets is None:
            external_stylesheets = [dbc.themes.BOOTSTRAP]
        super().__init__(
            title=title, external_stylesheets=external_stylesheets, **kwargs
        )
        self.full_results = results  # all xps event empty
        self.results = results  # without empty xps  (after preprocessing)
        self.preprocess_data()
        self.create_layout()
        self.load_callbacks()

    def preprocess_data(self):
        results = self.full_results
        assert results is not None
        # normalization + add new metrics
        self.config_store = ConfigStore()
        normalize_results(results, config_store=self.config_store)
        compute_extra_metrics(results)
        # keep only non-empty dataframes
        results = drop_empty_results(results)
        self.results = results
        # list available configs, instances, and metrics
        self.configs = sorted(extract_configs(results))
        self.instances = sorted(extract_instances(results))
        self.metrics = sorted(extract_metrics(results))
        self.full_configs = sorted(extract_configs(self.full_results))
        self.full_instances = sorted(extract_instances(self.full_results))
        self.full_metrics = sorted(extract_metrics(self.full_results))
        self.instances_with_sol_by_config = extract_instances_with_sol_by_config(
            self.full_results
        )
        # precompute aggregated data
        self.results_by_config = aggregate_results_by_config(
            results=results, configs=self.configs
        )
        self.nb_xps_by_config = extract_nb_xps_by_config(self.full_results)
        self.nbsolvedinstances_by_config = extract_nbsolvedinstances_by_config(
            results=results
        )
        self.percentsolvedinstances_by_config = (
            convert_nb2percentage_solvedinstances_by_config(
                nbsolvedinstances_by_config=self.nbsolvedinstances_by_config,
                n_xps_by_config=self.nb_xps_by_config,
            )
        )
        self.nbsolvedinstances_wo_status_by_config = (
            extract_nbsolvedinstances_wo_status_by_config(results=results)
        )
        self.percentsolvedinstances_wo_status_by_config = (
            convert_nb2percentage_solvedinstances_by_config(
                nbsolvedinstances_by_config=self.nbsolvedinstances_wo_status_by_config,
                n_xps_by_config=self.nb_xps_by_config,
            )
        )
        self.empty_xps_metadata = extract_empty_xps_metadata(self.full_results)
        self.df_best_metric_by_xp = compute_best_metrics_by_xp(
            results=self.full_results, metrics=self.full_metrics
        )
        self.df_convergence_time_by_xp = compute_convergence_time_by_xp(
            results=self.full_results, metrics=self.full_metrics
        )

    def create_layout(self):
        filter_style = {"margin-bottom": "1em"}
        controls = dbc.Card(
            [
                dbc.CardHeader([html.H2("Filters")]),
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                dbc.Switch(
                                    label=TIME_LOGSCALE_KEY,
                                    value=False,
                                    id=TIME_LOGSCALE_ID,
                                ),
                            ],
                            id=TIME_LOGSCALE_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Switch(
                                    label=Y_LOGSCALE_KEY,
                                    value=False,
                                    id=Y_LOGSCALE_ID,
                                ),
                            ],
                            id=Y_LOGSCALE_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Configs"),
                                dcc.Dropdown(
                                    self.full_configs,
                                    self.full_configs,
                                    multi=True,
                                    id=CONFIGS_ID,
                                ),
                            ],
                            id=CONFIGS_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Instances"),
                                dcc.Dropdown(
                                    ALIASES_INSTANCES + self.full_instances,
                                    ALIAS_INSTANCES_ALL,
                                    multi=True,
                                    id=INSTANCES_ID,
                                ),
                            ],
                            id=INSTANCES_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Metric"),
                                dcc.Dropdown(self.metrics, FIT, id=METRIC_ID),
                            ],
                            id=METRIC_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Clip at"),
                                dbc.Input(
                                    value=1e50, id=CLIP_ID, type="number", min=0.0
                                ),
                            ],
                            id=CLIP_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Aggregate with"),
                                dcc.Dropdown(
                                    sorted(map_stat_key2func_df.keys()),
                                    MEAN,
                                    id=STAT_ID,
                                ),
                            ],
                            id=STAT_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Quantile order"),
                                dbc.Input(
                                    min=0,
                                    max=1,
                                    value=0.5,
                                    step=0.05,
                                    type="number",
                                    id=Q_ID,
                                ),
                            ],
                            id=Q_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Min. proportion of data for aggregation"),
                                dcc.Slider(
                                    min=0.0,
                                    max=1.0,
                                    step=0.05,
                                    value=1.0,
                                    marks={i / 100: f"{i}%" for i in (0, 50, 100)},
                                    id=MIN_XP_PROP_ID,
                                ),
                            ],
                            id=MIN_XP_PROP_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Include solved xps without proof"),
                                dbc.Switch(
                                    value=False,
                                    id=SOLVED_WO_PROOF_ID,
                                    label=SOLVED_WO_PROOF_KEY,
                                ),
                            ],
                            id=SOLVED_WO_PROOF_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Optimization sense"),
                                dbc.Switch(
                                    value=False, id=MINIMIZING_ID, label=MINIMIZING_KEY
                                ),
                            ],
                            id=MINIMIZING_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Distance axis"),
                                dcc.Dropdown(
                                    options=[DIST_TO_BEST_REL, DIST_TO_BEST],
                                    value=DIST_TO_BEST_REL,
                                    id=SOLVERS_COMPETITION_DIST_ID,
                                ),
                            ],
                            id=SOLVERS_COMPETITION_DIST_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Time axis"),
                                dcc.Dropdown(
                                    options=[CONVERGENCE_TIME],
                                    value=CONVERGENCE_TIME,
                                    id=SOLVERS_COMPETITION_TIME_ID,
                                ),
                            ],
                            id=SOLVERS_COMPETITION_TIME_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Display all xps"),
                                dbc.Switch(
                                    value=True,
                                    id=SOLVERS_COMPETITION_ALL_XPS_ID,
                                    label=SOLVERS_COMPETITION_ALL_XPS_KEY,
                                ),
                            ],
                            id=SOLVERS_COMPETITION_ALL_XPS_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Transpose axes"),
                                dbc.Switch(
                                    label=TRANSPOSE_KEY,
                                    value=False,
                                    id=TRANSPOSE_ID,
                                ),
                            ],
                            id=TRANSPOSE_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Config"),
                                dcc.Dropdown(
                                    self.full_configs,
                                    value=self.full_configs[0],
                                    multi=False,
                                    id=CONFIG_ID,
                                ),
                            ],
                            id=CONFIG_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Instance"),
                                dcc.Dropdown(
                                    self.full_instances,
                                    value=self.full_instances[0],
                                    multi=False,
                                    id=INSTANCE_ID,
                                ),
                            ],
                            id=INSTANCE_DIV_ID,
                            style=filter_style,
                        ),
                        html.Div(
                            [
                                dbc.Label("Attempt"),
                                dcc.Dropdown(
                                    multi=False,
                                    id=RUN_ID,
                                ),
                            ],
                            id=RUN_DIV_ID,
                            style=filter_style,
                        ),
                    ],
                ),
            ]
        )

        graph_metric = dcc.Graph(id=GRAPH_METRIC_ID)

        graph_agg_metric = dcc.Graph(id=GRAPH_AGG_METRIC_ID)
        table_agg_metric = dash_table.DataTable(
            id=TABLE_AGG_METRIC_ID, **_get_dash_table_kwargs()
        )

        graph_nb_solved_instances = dcc.Graph(id=GRAPH_NB_SOLVED_INSTANCES_ID)
        table_nb_solved_instances = dash_table.DataTable(
            id=TABLE_NB_SOLVED_INSTANCES_ID, **_get_dash_table_kwargs()
        )

        graph_solvers_competition = dcc.Graph(id=GRAPH_SOLVERS_COMPETITION_ID)
        table_agg_rank = dash_table.DataTable(
            id=TABLE_AGG_RANK_ID, **_get_dash_table_kwargs()
        )
        table_agg_rank_explanation = dcc.Markdown(
            _explanation_table_agg_rank, className="mt-3"
        )

        config_explorer = dcc.Markdown(id=CONFIG_MD_ID)

        table_xp_status = html.P(id=TABLE_XP_STATUS_ID)
        table_xp = dash_table.DataTable(id=TABLE_XP_ID, **_get_dash_table_kwargs())
        table_xp_nodata = html.P("no data", id=TABLE_XP_NO_DATA_ID)

        table_empty_xps = dash_table.DataTable(
            data=_extract_dash_table_data_from_df(self.empty_xps_metadata),
            columns=_extract_dash_table_columns_from_df(
                self.empty_xps_metadata, numeric_columns=[]
            ),
            id=TABLE_EMPTY_XPS_ID,
            **_get_dash_table_kwargs(
                style_data={  # wrap long columns
                    "whiteSpace": "normal",
                    "height": "auto",
                }
            ),
        )

        graph_tabs = dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.H2("Graphs and Tables"),
                    ]
                ),
                dbc.CardBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    children=graph_metric,
                                    label="Metric evolution",
                                    tab_id=TAB_METRIC_ID,
                                ),
                                dbc.Tab(
                                    children=[
                                        graph_agg_metric,
                                        table_agg_metric,
                                    ],
                                    label="Metric aggregation along instances",
                                    tab_id=TAB_AGG_METRIC_ID,
                                ),
                                dbc.Tab(
                                    children=[
                                        graph_nb_solved_instances,
                                        table_nb_solved_instances,
                                    ],
                                    label="Nb of solved instances",
                                    tab_id=TAB_NB_SOLVED_INSTANCES_ID,
                                ),
                                dbc.Tab(
                                    children=[
                                        graph_solvers_competition,
                                        table_agg_rank,
                                        table_agg_rank_explanation,
                                    ],
                                    label="Solvers competition",
                                    tab_id=TAB_AGG_RANK_ID,
                                ),
                                dbc.Tab(
                                    children=config_explorer,
                                    label="Config explorer",
                                    tab_id=TAB_CONFIG_ID,
                                ),
                                dbc.Tab(
                                    children=html.Div(
                                        [table_xp_status, table_xp, table_xp_nodata],
                                        className="mt-3",
                                    ),
                                    label="Experiment data",
                                    tab_id=TAB_XP_ID,
                                ),
                                dbc.Tab(
                                    children=html.Div(
                                        [table_empty_xps],
                                        className="mt-3",
                                    ),
                                    label="Empty experiments",
                                    tab_id=TAB_EMPTY_XPS_ID,
                                ),
                            ],
                            id=TABS_ID,
                        )
                    ]
                ),
            ]
        )

        self.layout = dbc.Container(
            [
                html.H1(children="Discrete-Optimization Experiments Dashboard"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(controls, md=4, xl=3),
                        dbc.Col(
                            # [
                            #     html.H2("Graph"),
                            #     dbc.Row(dbc.Col(dcc.Graph(id="graph-metric-evolution"))),
                            #     dbc.Row(dbc.Col(dcc.Graph(id="graph-agg-metric-along-instances-evolution"))),
                            #     dbc.Row(dbc.Col(dcc.Graph(id="graph-nb-solved-instances-evolution"))),
                            # ],
                            graph_tabs,
                            md=8,
                            xl=9,
                        ),
                    ],
                ),
            ],
            fluid=False,
            style={"$container-max-widths": "(  xl: 10820px,);"},
        )

    def load_callbacks(self):
        # Graph graph-metric-evolution: 1 config x 1 instance => evolution of a metric
        @self.callback(
            Output(component_id=GRAPH_METRIC_ID, component_property="figure"),
            inputs=dict(
                configs=Input(component_id=CONFIGS_ID, component_property="value"),
                instances=Input(component_id=INSTANCES_ID, component_property="value"),
                metric=Input(component_id=METRIC_ID, component_property="value"),
                clip_value=Input(component_id=CLIP_ID, component_property="value"),
                time_log_scale=Input(
                    component_id=TIME_LOGSCALE_ID, component_property="value"
                ),
                metric_log_scale=Input(
                    component_id=Y_LOGSCALE_ID, component_property="value"
                ),
            ),
        )
        def update_graph_metric(
            configs, instances, metric, time_log_scale, metric_log_scale, clip_value
        ):
            instances = self._replace_instances_aliases(
                instances, configs=configs
            )  # interpret @all alias
            results = clip_results(
                filter_results(self.results, configs=configs, instances=instances),
                clip_value=clip_value,
            )
            map_xp2metric = {
                get_experiment_name(df, with_run_nb=has_multiple_runs(results)): df[
                    metric
                ]
                for df in results
                if metric in df
            }
            return create_graph_from_series_dict(
                map_label2ser=map_xp2metric,
                time_log_scale=time_log_scale,
                y_log_scale=metric_log_scale,
                legend_title="Experiments",
            )

        # Graph graph-agg-metric-along-instance-evolution: 1 config => aggregation of a metric along instances
        @self.callback(
            output=dict(
                plot=Output(
                    component_id=GRAPH_AGG_METRIC_ID,
                    component_property="figure",
                ),
                data=Output(
                    component_id=TABLE_AGG_METRIC_ID,
                    component_property="data",
                ),
                columns=Output(
                    component_id=TABLE_AGG_METRIC_ID,
                    component_property="columns",
                ),
            ),
            inputs=dict(
                configs=Input(component_id=CONFIGS_ID, component_property="value"),
                instances=Input(component_id=INSTANCES_ID, component_property="value"),
                stat=Input(component_id=STAT_ID, component_property="value"),
                metric=Input(component_id=METRIC_ID, component_property="value"),
                q=Input(component_id=Q_ID, component_property="value"),
                clip_value=Input(component_id=CLIP_ID, component_property="value"),
                time_log_scale=Input(
                    component_id=TIME_LOGSCALE_ID, component_property="value"
                ),
                metric_log_scale=Input(
                    component_id=Y_LOGSCALE_ID, component_property="value"
                ),
                min_xp_proportion=Input(
                    component_id=MIN_XP_PROP_ID, component_property="value"
                ),
            ),
        )
        def update_graph_agg_metric(
            configs,
            instances,
            stat,
            metric,
            q,
            time_log_scale,
            clip_value,
            min_xp_proportion,
            metric_log_scale,
        ):
            instances = self._replace_instances_aliases(
                instances, configs=configs
            )  # interpret @all alias
            # filter and clip
            results_by_config = {
                config: clip_df(df, clip_value=clip_value)
                for config, df in self.results_by_config.items()
                if config in configs
            }
            stat_by_config = compute_stat_by_config(
                results_by_config=results_by_config,
                stat=stat,
                q=q,
                instances=instances,
                min_xp_proportion=min_xp_proportion,
            )
            stat_metric_by_config = {
                config: df[metric]
                for config, df in stat_by_config.items()
                if metric in df
            }
            (
                nb_xps_by_config,
                nb_xps_wo_sol_by_config,
            ) = extract_nb_xps_w_n_wo_sol_by_config(
                results=self.full_results, configs=configs, instances=instances
            )
            df_summary = construct_summary_metric_agg(
                stat_by_config=stat_by_config,
                configs=configs,
                nb_xps_by_config=nb_xps_by_config,
                nb_xps_wo_sol_by_config=nb_xps_wo_sol_by_config,
            )
            return dict(
                plot=create_graph_from_series_dict(
                    map_label2ser=stat_metric_by_config,
                    time_log_scale=time_log_scale,
                    legend_title="Configs",
                    y_log_scale=metric_log_scale,
                ),
                data=_extract_dash_table_data_from_df(df_summary),
                columns=_extract_dash_table_columns_from_df(
                    df_summary, non_numeric_columns=[CONFIG]
                ),
            )

        # Graph graph-nb-solved-instances-evolution: 1 config => nb of instances solved depending on time
        @self.callback(
            output=dict(
                plot=Output(
                    component_id=GRAPH_NB_SOLVED_INSTANCES_ID,
                    component_property="figure",
                ),
                data=Output(
                    component_id=TABLE_NB_SOLVED_INSTANCES_ID, component_property="data"
                ),
                columns=Output(
                    component_id=TABLE_NB_SOLVED_INSTANCES_ID,
                    component_property="columns",
                ),
            ),
            inputs=dict(
                configs=Input(component_id=CONFIGS_ID, component_property="value"),
                time_log_scale=Input(
                    component_id=TIME_LOGSCALE_ID, component_property="value"
                ),
                transpose=Input(component_id=TRANSPOSE_ID, component_property="value"),
                include_solved_wo_proof=Input(
                    component_id=SOLVED_WO_PROOF_ID, component_property="value"
                ),
            ),
        )
        def update_graph_nb_solved_instances(
            configs, time_log_scale, transpose, include_solved_wo_proof
        ):
            if include_solved_wo_proof:
                percentsolvedinstances_by_config = (
                    self.percentsolvedinstances_wo_status_by_config
                )
                nbsolvedinstances_by_config = self.nbsolvedinstances_wo_status_by_config
            else:
                percentsolvedinstances_by_config = self.percentsolvedinstances_by_config
                nbsolvedinstances_by_config = self.nbsolvedinstances_by_config
            percentsolvedinstances_by_config = {
                config: ser
                for config, ser in percentsolvedinstances_by_config.items()
                if config in configs
            }
            df_summary = construct_summary_nbsolved_instances(
                nbsolvedinstances_by_config=nbsolvedinstances_by_config,
                nb_xps_by_config=self.nb_xps_by_config,
                configs=configs,
            )
            return dict(
                plot=create_graph_from_series_dict(
                    map_label2ser=percentsolvedinstances_by_config,
                    time_log_scale=time_log_scale,
                    legend_title="Configs",
                    transpose=transpose,
                ),
                data=_extract_dash_table_data_from_df(df_summary),
                columns=_extract_dash_table_columns_from_df(
                    df_summary, non_numeric_columns=[CONFIG]
                ),
            )

        # Table aggreated ranks + dist to best metric
        @self.callback(
            output=Output(
                component_id=MINIMIZING_ID,
                component_property="value",
            ),
            inputs=Input(component_id=METRIC_ID, component_property="value"),
        )
        def update_minimizing_button(metric):
            if metric == "fit":
                return False
            else:
                return True

        @self.callback(
            output=dict(
                data=Output(
                    component_id=TABLE_AGG_RANK_ID,
                    component_property="data",
                ),
                columns=Output(
                    component_id=TABLE_AGG_RANK_ID,
                    component_property="columns",
                ),
                plot=Output(
                    component_id=GRAPH_SOLVERS_COMPETITION_ID,
                    component_property="figure",
                ),
            ),
            inputs=dict(
                configs=Input(component_id=CONFIGS_ID, component_property="value"),
                instances=Input(component_id=INSTANCES_ID, component_property="value"),
                stat=Input(component_id=STAT_ID, component_property="value"),
                metric=Input(component_id=METRIC_ID, component_property="value"),
                q=Input(component_id=Q_ID, component_property="value"),
                clip_value=Input(component_id=CLIP_ID, component_property="value"),
                minimizing=Input(
                    component_id=MINIMIZING_ID, component_property="value"
                ),
                time_log_scale=Input(
                    component_id=TIME_LOGSCALE_ID, component_property="value"
                ),
                transpose=Input(component_id=TRANSPOSE_ID, component_property="value"),
                all_xps=Input(
                    component_id=SOLVERS_COMPETITION_ALL_XPS_ID,
                    component_property="value",
                ),
                time_label=Input(
                    component_id=SOLVERS_COMPETITION_TIME_ID, component_property="value"
                ),
                dist_label=Input(
                    component_id=SOLVERS_COMPETITION_DIST_ID, component_property="value"
                ),
            ),
        )
        def update_table_rank_agg(
            configs,
            instances,
            stat,
            metric,
            q,
            clip_value,
            minimizing,
            time_log_scale,
            transpose,
            dist_label,
            time_label,
            all_xps,
        ):
            instances = self._replace_instances_aliases(
                instances, configs=configs
            )  # interpret @all alias
            # clip
            df_best_metric_by_xp = clip_df(
                self.df_best_metric_by_xp, clip_value=clip_value
            )
            # compute dataframe
            df_summary, df_by_xp = compute_summary_agg_ranks_and_dist_to_best_metric(
                df_best_metric_by_xp=df_best_metric_by_xp,
                df_convergence_time_by_xp=self.df_convergence_time_by_xp,
                configs=configs,
                instances=instances,
                metric=metric,
                stat=stat,
                q=q,
                minimizing=minimizing,
            )
            # create graph
            if not all_xps:
                df_by_xp = None
            plot = create_solvers_competition_graph(
                df_summary=df_summary,
                df_by_xp=df_by_xp,
                x=time_label,
                y=dist_label,
                z=CONFIG,
                legend_title="Configs",
                time_log_scale=time_log_scale,
                transpose=transpose,
            )
            return dict(
                data=_extract_dash_table_data_from_df(df_summary),
                columns=_extract_dash_table_columns_from_df(
                    df_summary, non_numeric_columns=[CONFIG]
                ),
                plot=plot,
            )

        # Config explorer
        @self.callback(
            Output(component_id=CONFIG_MD_ID, component_property="children"),
            inputs=dict(
                config_name=Input(component_id=CONFIG_ID, component_property="value"),
            ),
        )
        def update_config_display(config_name: str) -> str:
            configs = self.config_store.get_configs(config_name)
            config_str = "\n\n".join(json.dumps(config, indent=4) for config in configs)
            md_str = f"```python\n{config_str}\n```\n"
            if len(configs) > 1:
                md_str = (
                    f"> WARNING: {len(configs)} configs with same name found!\n\n"
                    + md_str
                )
            elif len(configs) == 0:
                md_str = f"> WARNING: no config found with given name!\n"
            return md_str

        # Xp explorer
        @self.callback(
            output=dict(
                options=Output(component_id=RUN_ID, component_property="options"),
                value=Output(component_id=RUN_ID, component_property="value"),
            ),
            inputs=dict(
                config=Input(component_id=CONFIG_ID, component_property="value"),
                instance=Input(component_id=INSTANCE_ID, component_property="value"),
            ),
        )
        def update_run_options(config: str, instance: str) -> dict[str, Any]:
            nb_results = len(
                filter_results(
                    results=self.full_results, configs=[config], instances=[instance]
                )
            )
            return dict(
                options=list(range(nb_results)),
                value=0,
            )

        @self.callback(
            output=dict(
                status=Output(
                    component_id=TABLE_XP_STATUS_ID, component_property="children"
                ),
                data=Output(component_id=TABLE_XP_ID, component_property="data"),
                columns=Output(component_id=TABLE_XP_ID, component_property="columns"),
                nodata=Output(
                    component_id=TABLE_XP_NO_DATA_ID, component_property="className"
                ),
            ),
            inputs=dict(
                config=Input(component_id=CONFIG_ID, component_property="value"),
                instance=Input(component_id=INSTANCE_ID, component_property="value"),
                run=Input(component_id=RUN_ID, component_property="value"),
            ),
        )
        def update_xp_data(config: str, instance: str, run: Optional[int]) -> Any:
            if run is None:
                return dict(
                    data=[], columns=[], nodata=_convert_bool2classname(True), status=""
                )
            df = filter_results(
                results=self.full_results, configs=[config], instances=[instance]
            )[run]
            status = get_status_str(df)
            df = df.reset_index()
            return dict(
                data=_extract_dash_table_data_from_df(df),
                columns=_extract_dash_table_columns_from_df(df),
                nodata=_convert_bool2classname(len(df) == 0),
                status=f"Status: {status}",
            )

        # Filters disabling
        @self.callback(
            output=dict(
                metric=Output(
                    component_id=METRIC_DIV_ID, component_property="className"
                ),
                configs=Output(
                    component_id=CONFIGS_DIV_ID, component_property="className"
                ),
                instances=Output(
                    component_id=INSTANCES_DIV_ID, component_property="className"
                ),
                stat=Output(component_id=STAT_DIV_ID, component_property="className"),
                q=Output(component_id=Q_DIV_ID, component_property="className"),
                transpose=Output(
                    component_id=TRANSPOSE_DIV_ID, component_property="className"
                ),
                clip=Output(component_id=CLIP_DIV_ID, component_property="className"),
                config=Output(
                    component_id=CONFIG_DIV_ID, component_property="className"
                ),
                instance=Output(
                    component_id=INSTANCE_DIV_ID, component_property="className"
                ),
                run=Output(component_id=RUN_DIV_ID, component_property="className"),
                time_log_scale=Output(
                    component_id=TIME_LOGSCALE_DIV_ID, component_property="className"
                ),
                metric_log_scale=Output(
                    component_id=Y_LOGSCALE_ID, component_property="className"
                ),
                min_xp_prop=Output(
                    component_id=MIN_XP_PROP_DIV_ID, component_property="className"
                ),
                minimizing=Output(
                    component_id=MINIMIZING_DIV_ID, component_property="className"
                ),
                solved_wo_proof=Output(
                    component_id=SOLVED_WO_PROOF_DIV_ID, component_property="className"
                ),
                solvers_competition_all_xps=Output(
                    component_id=SOLVERS_COMPETITION_ALL_XPS_DIV_ID,
                    component_property="className",
                ),
                solvers_competition_dist=Output(
                    component_id=SOLVERS_COMPETITION_DIST_DIV_ID,
                    component_property="className",
                ),
                solvers_competition_time=Output(
                    component_id=SOLVERS_COMPETITION_TIME_DIV_ID,
                    component_property="className",
                ),
            ),
            inputs=dict(
                active_tab=Input(component_id=TABS_ID, component_property="active_tab"),
                stat=Input(component_id=STAT_ID, component_property="value"),
            ),
        )
        def update_filters(active_tab: str, stat: str):
            if active_tab == TAB_METRIC_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=True,
                        metric_log_scale=True,
                        min_xp_prop=False,
                        metric=True,
                        configs=True,
                        instances=True,
                        stat=False,
                        q=False,
                        transpose=False,
                        clip=True,
                        config=False,
                        instance=False,
                        run=False,
                        minimizing=False,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_AGG_METRIC_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=True,
                        metric_log_scale=True,
                        min_xp_prop=True,
                        metric=True,
                        configs=True,
                        instances=True,
                        stat=True,
                        q=stat == QUANTILE,
                        transpose=False,
                        clip=True,
                        config=False,
                        instance=False,
                        run=False,
                        minimizing=False,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_AGG_RANK_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=True,
                        metric_log_scale=False,
                        min_xp_prop=False,
                        metric=True,
                        configs=True,
                        instances=True,
                        stat=True,
                        q=stat == QUANTILE,
                        transpose=True,
                        clip=True,
                        config=False,
                        instance=False,
                        run=False,
                        minimizing=True,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=True,
                        solvers_competition_dist=True,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_NB_SOLVED_INSTANCES_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=True,
                        metric_log_scale=False,
                        min_xp_prop=False,
                        metric=False,
                        configs=True,
                        instances=False,
                        stat=False,
                        q=False,
                        transpose=True,
                        clip=False,
                        config=False,
                        instance=False,
                        run=False,
                        minimizing=False,
                        solved_wo_proof=True,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_CONFIG_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=False,
                        metric_log_scale=False,
                        min_xp_prop=False,
                        metric=False,
                        configs=False,
                        instances=False,
                        stat=False,
                        q=False,
                        transpose=False,
                        clip=False,
                        config=True,
                        instance=False,
                        run=False,
                        minimizing=False,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_XP_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=False,
                        metric_log_scale=False,
                        min_xp_prop=False,
                        metric=False,
                        configs=False,
                        instances=False,
                        stat=False,
                        q=False,
                        transpose=False,
                        clip=False,
                        config=True,
                        instance=True,
                        run=True,
                        minimizing=False,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            elif active_tab == TAB_EMPTY_XPS_ID:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=False,
                        metric_log_scale=False,
                        min_xp_prop=False,
                        metric=False,
                        configs=False,
                        instances=False,
                        stat=False,
                        q=False,
                        transpose=False,
                        clip=False,
                        config=False,
                        instance=False,
                        run=False,
                        minimizing=False,
                        solved_wo_proof=False,
                        solvers_competition_all_xps=False,
                        solvers_competition_dist=False,
                        solvers_competition_time=False,
                    )
                )
            else:
                return _convert_bool2classname_dict(
                    dict(
                        time_log_scale=True,
                        metric_log_scale=True,
                        min_xp_prop=True,
                        metric=True,
                        configs=True,
                        instances=True,
                        stat=True,
                        q=True,
                        transpose=True,
                        clip=True,
                        config=True,
                        instance=True,
                        run=True,
                        minimizing=True,
                        solved_wo_proof=True,
                        solvers_competition_all_xps=True,
                        solvers_competition_dist=True,
                        solvers_competition_time=True,
                    )
                )

        # store callbacks for unit testing
        self.update_filters = update_filters
        self.update_xp_data = update_xp_data
        self.update_run_options = update_run_options
        self.update_graph_agg_metric = update_graph_agg_metric
        self.update_graph_metric = update_graph_metric
        self.update_table_rank_agg = update_table_rank_agg
        self.update_config_display = update_config_display
        self.update_graph_nb_solved_instances = update_graph_nb_solved_instances

    def _replace_instances_aliases(
        self, instances: list[str], configs: list[str]
    ) -> list[str]:
        if ALIAS_INSTANCES_ALL in instances:
            return self.full_instances
        if ALIAS_INSTANCES_WITHSOL in instances:
            instances_with_sol = set.intersection(
                *(self.instances_with_sol_by_config[config] for config in configs)
            )
            # replace @withsol, keeping instances order
            instances_with_sol_not_yet_in_instances = [
                instance
                for instance in sorted(instances_with_sol)
                if instance not in instances
            ]
            i = instances.index(ALIAS_INSTANCES_WITHSOL)
            instances_before = instances[:i]
            instances_after = instances[i + 1 :]
            instances = (
                instances_before
                + instances_with_sol_not_yet_in_instances
                + instances_after
            )

        return instances


def _convert_bool2classname_dict(d: dict[str, bool]) -> dict[str, str]:
    return {k: _convert_bool2classname(v) for k, v in d.items()}


def _convert_bool2classname(v: bool) -> str:
    return "d-block" if v else "d-none"


def _extract_dash_table_data_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict("records")


def _extract_dash_table_columns_from_df(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    non_numeric_columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    data_table_columns = []
    for c in df.columns:
        col = {"id": c, "name": c}
        if (
            numeric_columns is None
            and (non_numeric_columns is None or c not in non_numeric_columns)
        ) or (numeric_columns is not None and c in numeric_columns):
            col.update(_data_table_numeric_column_extras)
        data_table_columns.append(col)
    return data_table_columns


def _get_dash_table_kwargs(**kwargs):
    kwargs_table = dict(
        page_size=20,
        export_format="csv",
    )
    kwargs_table.update(kwargs)
    return kwargs_table


_data_table_numeric_column_extras = dict(type="numeric", format={"specifier": ".5"})

_explanation_table_agg_rank = """
In this table, given a metric,
we compute instance by instance:
- solver rank among all configs,
- distance to the best metric value among all configs (relative and absolute),
- convergence time (first time the last metric value is reached).

Then we aggregate along instances:
- by counting ranks #1, #2, ... and failed instances,
- by applying the chosen aggregation method (in the "filters" panel) on the distance and on the convergence time.

The "minimizing" switch specifies whether the metric is supposed to be minimized or maximized.

NB: in the case where several attempts were made for a choice "config x instance", before computing rankings,
we first take the median metric among all those attempts, so that we got only one (at most) value by
"config x instance" tuple. Thus for each instance, the rankings will go at most up to the number of configs.
In case of tied metrics, the configs share the same rankings and the other rankings will be degraded accordingly
(e.g. #1, #1, #3, #4).
"""
