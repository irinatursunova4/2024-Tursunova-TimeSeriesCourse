import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config


def compute_mp(ts1, m, exclusion_zone=None, ts2=None):
    """
    Compute the matrix profile between two time series or within one time series.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    m : int
        The subsequence length.

    exclusion_zone : int, default = None
        Exclusion zone.

    ts2 : numpy.ndarray, default = None
        The second time series.

    Returns
    -------
    output : dict
        The matrix profile structure
        (matrix profile, matrix profile index, subsequence length,
        exclusion zone, the first and second time series).
    """
    ts1 = ts1.flatten()
    # Удаляем NaN из ts1
    ts1 = ts1[~np.isnan(ts1)]

    if ts2 is not None:
        ts2 = ts2.flatten()
        # Удаляем NaN из ts2
        ts2 = ts2[~np.isnan(ts2)]

    # Проверяем, что длина временных рядов достаточна для размера окна m
    if len(ts1) < m or (ts2 is not None and len(ts2) < m):
        raise ValueError("Длина временных рядов меньше размера окна m после удаления NaN.")

    if exclusion_zone:
        config.STUMPY_EXCL_ZONE_DENOM = exclusion_zone

    # Вычисляем матричный профиль
    if ts2 is not None:
        mp = stumpy.stump(ts1, m=m, T_B=ts2)
    else:
        mp = stumpy.stump(ts1, m=m)

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm': m,
            'excl_zone': exclusion_zone,
            'data': {'ts1': ts1, 'ts2': ts2}
            }