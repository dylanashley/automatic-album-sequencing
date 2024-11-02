# -*- coding: ascii -*-

import collections
from features import compute_features, get_duration
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import plotly.graph_objects as go
import scipy.interpolate
import seaborn as sns
import torch
from typing import Callable, Optional


def _plot_matplotlib(
    labels: list[str],
    values: list[float],
    min_value: float,
    max_value: float,
    template: Optional[Callable[[float], float]] = None,
    basename: bool = False,
) -> plt.Figure:
    """Draws a narrative essence plot for a fitting using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 6))
    with sns.axes_style("ticks"), sns.color_palette("colorblind"):
        x = np.linspace(0, 1, num=1000)
        if template is not None:
            ax.plot(
                [scale(i, 0, 1, 1, len(labels)) for i in x],
                [scale(template(i), 0, 1, min_value, max_value) for i in x],
            )
        ax.plot(
            np.arange(len(labels)) + 1,
            [scale(values[song], 0, 1, min_value, max_value) for song in labels],
            "o-",
        )
        ax.set_xticks(np.arange(len(labels)) + 1)
        xticklabels = [
            os.path.basename(label) if basename else label for label in labels
        ]
        ax.set_xticklabels(xticklabels, rotation=270)
        ax.set_ylabel("Narrative Essence", labelpad=5)
    fig.tight_layout()
    return fig


def _plot_plotly(
    labels: list[str],
    values: list[float],
    min_value: float,
    max_value: float,
    template: Optional[Callable[[float], float]] = None,
    basename: bool = False,
) -> go.Figure:
    """Draws a narrative essence plot for a fitting using plotly."""
    fig = go.Figure()
    x = np.linspace(0, 1, num=1000)
    if template is not None:
        fig.add_trace(
            go.Scatter(
                x=[scale(i, 0, 1, 1, len(labels)) for i in x],
                y=[scale(template(i), 0, 1, min_value, max_value) for i in x],
                mode="lines",
                name="Template",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(labels)) + 1,
            y=[scale(values[song], 0, 1, min_value, max_value) for song in labels],
            mode="markers+lines",
            name="Fitting",
        )
    )
    fig.update_layout(
        xaxis=dict(
            tickvals=np.arange(len(labels)) + 1,
            ticktext=[
                os.path.basename(label) if basename else label for label in labels
            ],
            tickangle=-90,
        ),
        yaxis=dict(title="Narrative Essence"),
    )
    return fig


def build_template(
    xy: list[tuple[float, float]],
) -> Callable[[float], float]:
    """Returns a callable narrative template curve."""
    x, y = zip(*xy)
    min_y = min(y)
    max_y = max(y)
    y = list(y)
    if min_y != max_y:
        for i in range(len(y)):
            y[i] = scale(y[i], min_y, max_y, 0, 1)
    if len(x) == 1:
        kind = "nearest"
    elif len(x) == 2:
        kind = "linear"
    elif len(x) == 3:
        kind = "quadratic"
    else:
        kind = "cubic"
    return scipy.interpolate.interp1d(x, y, kind=kind)


def default_template(
    time: float,
    order: int = 2,
) -> float:
    """Returns the value of a narrative template curve for a time in the
    range [0, 1].
    """
    assert 0 <= time <= 1
    assert order in [1, 2]
    if order == 1:
        if time <= 0.2:
            return 1 / 2 + 5 / 4 * time
        elif time <= 0.5:
            return 5 / 4 - 5 / 2 * time
        elif time <= 0.8:
            return -5 / 3 + 10 / 3 * time
        else:
            return 2 - 5 / 4 * time
    else:
        if time <= 0.2:
            return 1 / 2 + 5 / 2 * time - 25 / 4 * time**2
        elif time <= 0.3:
            return -1 / 4 + 10 * time - 25 * time**2
        elif time <= 0.5:
            return 25 / 8 - 25 / 2 * time + 25 / 2 * time**2
        elif time <= 0.65:
            return 50 / 9 - 200 / 9 * time + 200 / 9 * time**2
        elif time <= 0.8:
            return -119 / 9 + 320 / 9 * time - 200 / 9 * time**2
        else:
            return -3 + 10 * time - 25 / 4 * time**2


def fit_values(
    values: dict[str, float],
    template: Callable[[float], float] = default_template,
) -> tuple[list[str], float]:
    """Fits a set of values to a narrative template curve."""
    values = collections.OrderedDict(values)
    filenames = list(values.keys())
    rv = [-1 for _ in range(len(values))]
    distances = np.zeros((len(values), len(values)), dtype=float)
    for i, y in enumerate(values.values()):
        for j in range(len(values)):
            x = scale(j, 0, len(values), 0, 1)
            distances[i, j] = np.square(y - template(x))

    # binary search to find smallest deviation matching
    candidates = np.sort(distances.flatten())
    min_idx = 0
    max_idx = len(candidates) - 1
    while min_idx != max_idx:
        pivot = min_idx + (max_idx - min_idx) // 2
        edges = [
            (filenames[i], j)
            for i in range(len(values))
            for j in range(len(values))
            if distances[i, j] <= candidates[pivot]
        ]
        graph = nx.Graph(edges)
        try:
            matching = nx.bipartite.maximum_matching(graph)
            if all([filename in matching for filename in filenames]):
                max_idx = pivot
            else:
                min_idx = pivot + 1
        except nx.AmbiguousSolution:
            min_idx = pivot + 1

    # now minimize the average devation as well
    edges = [
        (filenames[i], j, {"weight": distances[i, j]})
        for i in range(len(values))
        for j in range(len(values))
        if distances[i, j] <= candidates[max_idx]
    ]
    graph = nx.Graph(edges)
    matching = nx.bipartite.minimum_weight_full_matching(graph)

    # get the total deviation
    total_devation = 0
    for start, end, properties in edges:
        if matching[start] == end:
            total_devation += properties["weight"]

    return [matching[i] for i in range(len(filenames))], total_devation


def get_value(
    filename: str,
    encoder: torch.nn.Module,
) -> float:
    """Returns the narrative essence for an audio file."""
    a = torch.from_numpy(compute_features(filename).to_numpy()).float()
    durations_mean = 257.6875
    durations_std = 191.8393
    duration = get_duration(filename)
    duration = (duration - durations_mean) / durations_std
    duration = torch.FloatTensor([duration])
    features = torch.cat([a, duration.repeat(7)]).unsqueeze(0)
    value = encoder(features)
    return value.item()


def plot(
    labels: list[str],
    values: list[float],
    min_value: float = 0,
    max_value: float = 1,
    template: Optional[Callable[[float], float]] = None,
    basename: bool = False,
    plotly: bool = False,
) -> plt.Figure | go.Figure:
    """Draws a narrative essence plot for a fitting."""
    if plotly:
        return _plot_plotly(
            labels,
            values,
            min_value,
            max_value,
            template=template,
            basename=basename,
        )
    else:
        return _plot_matplotlib(
            labels,
            values,
            min_value,
            max_value,
            template=template,
            basename=basename,
        )


def scale(
    value: float,
    start_min: float,
    start_max: float,
    end_min: float,
    end_max: float,
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)
