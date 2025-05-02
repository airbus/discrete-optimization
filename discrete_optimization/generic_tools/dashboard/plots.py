from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import plotly
except ImportError:
    plotly_available = False
else:
    plotly_available = True
    import plotly.graph_objects as go


logger = logging.getLogger(__name__)


def create_graph_from_series_dict(
    map_label2ser: dict[str, pd.Series],
    with_time_log_scale: bool = False,
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
    if with_time_log_scale:
        if transpose:
            fig.update_yaxes(type="log")
        else:
            fig.update_xaxes(type="log")
    return fig
