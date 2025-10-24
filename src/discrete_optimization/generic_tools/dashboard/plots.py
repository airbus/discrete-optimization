from __future__ import annotations

import logging
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly
except ImportError:
    plotly_available = False
else:
    plotly_available = True
    import plotly.express as px
    import plotly.graph_objects as go


logger = logging.getLogger(__name__)


def create_graph_from_series_dict(
    map_label2ser: dict[str, pd.Series],
    time_log_scale: bool = False,
    y_log_scale: bool = False,
    legend_title: str = "labels",
    transpose: bool = False,
) -> go.Figure:
    fig = go.Figure()
    x_label = ""
    y_label = ""
    for name, ser in map_label2ser.items():
        ser = ser.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ser) < 2:
            mode = "markers"
        else:
            mode = "lines"
        if transpose:
            y = ser.index
            x = ser
            y_label = ser.index.name
            x_label = ser.name
        else:
            x = ser.index
            y = ser
            x_label = ser.index.name
            y_label = ser.name
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode=mode))

    if len(map_label2ser) == 0:
        fig.add_annotation(
            text="NO DATA",
            font=dict(size=20),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    fig.update_layout(
        xaxis=dict(title=dict(text=x_label)),
        yaxis=dict(title=dict(text=y_label)),
        legend=dict(title=dict(text=legend_title)),
    )
    if time_log_scale:
        if transpose:
            fig.update_yaxes(type="log")
        else:
            fig.update_xaxes(type="log")
    if y_log_scale:
        if transpose:
            fig.update_xaxes(type="log")
        else:
            fig.update_yaxes(type="log")
    return fig


def create_solvers_competition_graph(
    df_summary: pd.DataFrame,
    df_by_xp: Optional[pd.DataFrame] = None,
    x: str = "convergence time (s)",
    y: str = "dist to best",
    z: str = "config",
    legend_title: Optional[str] = None,
    time_log_scale: bool = False,
    transpose: bool = False,
    opacity_by_instance: float = 0.5,
) -> go.Figure:
    if legend_title is None:
        legend_title = z

    # agg data
    if transpose:
        fig = px.scatter(df_summary, x=y, y=x, color=z)
    else:
        fig = px.scatter(df_summary, x=x, y=y, color=z)

    # by xp data
    if df_by_xp is not None:
        if transpose:
            hovertemplate_by_instance = (
                z
                + "=%{fullData.name}<br>"
                + "instance=%{text}<br>"
                + y
                + "=%{x}<br>"
                + x
                + "=%{y}"
                + "<extra></extra>"
            )
        else:
            hovertemplate_by_instance = (
                z
                + "=%{fullData.name}<br>"
                + "instance=%{text}<br>"
                + x
                + "=%{x}<br>"
                + y
                + "=%{y}"
                + "<extra></extra>"
            )
        for trace in fig.data:
            # agg data above, with fixed size, separated legend
            trace.legend = "legend2"
            trace.zorder = 1
            trace.marker.size = 7
            # by xp data: transparent and smaller
            marker = copy(trace.marker)
            marker.opacity = opacity_by_instance
            marker.size /= 1.5
            # xp data for given config
            config = trace.name
            df = df_by_xp.loc[(config, slice(None))]
            if transpose:
                xx = df[y]
                yy = df[x]
            else:
                xx = df[x]
                yy = df[y]
            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    text=df.index,
                    name=config,
                    marker=marker,
                    mode="markers",
                    legend="legend",
                    showlegend=True,
                    hovertemplate=hovertemplate_by_instance,
                )
            )
        fig.update_layout(
            legend2=dict(title=dict(text=f"{legend_title} agg"), tracegroupgap=0),
            legend=dict(
                title=dict(text=f"{legend_title} all xp"),
                tracegroupgap=0,
                y=0,
                yanchor="bottom",
            ),
        )
    else:
        fig.update_layout(
            legend=dict(title=legend_title),
        )
    if time_log_scale:
        if transpose:
            fig.update_yaxes(type="log")
        else:
            fig.update_xaxes(type="log")
    return fig
