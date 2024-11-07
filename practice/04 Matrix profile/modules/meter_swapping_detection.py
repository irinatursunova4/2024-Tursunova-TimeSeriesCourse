import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px

plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions, cutoff, house_idx):
    """
    Split time series into two parts: Head and Tail.

    Parameters
    ---------
    consumptions : dict
        Set of time series.

    cutoff : pandas.Timestamp
        Cut-off point.

    house_idx : list
        Indices of houses.

    Returns
    --------
    heads : dict
        Heads of time series.

    tails : dict
        Tails of time series.
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]

    return heads, tails


import numpy as np


def get_min_series_length(heads, tails):
    """
    Определяет минимальную длину среди всех временных рядов в heads и tails.
    """
    min_length = float('inf')
    for ts in list(heads.values()) + list(tails.values()):
        min_length = min(min_length, len(ts))
    return min_length


def meter_swapping_detection(heads, tails, house_idx, m):
    """
    Find the swapped time series pair with minimum swap-score.

    Parameters
    ---------
    heads : dict
        Heads of time series.

    tails : dict
        Tails of time series.

    house_idx : list
        Indices of houses.

    m : int
        Subsequence length.

    Returns
    --------
    min_score : dict
       Time series pair with minimum swap-score.
    """
    eps = 0.001
    min_score = {'score': float('inf'), 'i': None, 'j': None, 'mp_j': None}

    for i in house_idx:
        for j in house_idx:
            if i != j:
                head_ts = heads[f'H_{i}'].values.flatten()
                tail_ts = tails[f'T_{j}'].values.flatten()

                # Удаляем NaN
                head_ts = head_ts[~np.isnan(head_ts)]
                tail_ts = tail_ts[~np.isnan(tail_ts)]

                # Проверяем длину временных рядов после удаления NaN
                min_length = min(len(head_ts), len(tail_ts))
                if min_length < m:
                    print(f"Пропускаем пару (H_{i}, T_{j}) из-за недостаточной длины после удаления NaN.")
                    continue

                # Вычисляем матричный профиль между H_i и T_j
                mp_result = compute_mp(head_ts, m, ts2=tail_ts)
                head_to_tail_score = np.min(mp_result['mp'])

                # Вычисляем матричный профиль внутри H_i
                mp_result_self = compute_mp(head_ts, m)
                head_self_score = np.min(mp_result_self['mp'])

                # Вычисляем swap_score
                swap_score = head_to_tail_score / (head_self_score + eps)

                if swap_score < min_score['score']:
                    min_score = {
                        'score': swap_score,
                        'i': i,
                        'j': j,
                        'mp_j': mp_result['mp']
                    }

    return min_score


def plot_consumptions_ts(consumptions, cutoff, house_idx):
    """
    Plot a set of input time series and cutoff vertical line.

    Parameters
    ---------
    consumptions : dict
        Set of time series.

    cutoff : pandas.Timestamp
        Cut-off point.

    house_idx : list
        Indices of houses.
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:, 0],
                                 name=f"House {house_idx[i]}"), row=i + 1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red", row=i + 1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="white",
                      paper_bgcolor='white',
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="png")