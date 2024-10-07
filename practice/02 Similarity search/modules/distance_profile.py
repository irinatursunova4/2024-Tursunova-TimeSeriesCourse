import numpy as np

from utils import z_normalize
from metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    # INSERT YOUR CODE
        # Шаг 2: Проверка на нормализацию
    if is_normalize:
        query = z_normalize(query)

    # Шаг 4: Основной цикл по временном ряду
    for i in range(N):
        # Если нужно нормализовать, нормализуем текущую подпоследовательность
        if is_normalize:
            ts_i_m = z_normalize(ts[i:i + m])
        else:
            ts_i_m = ts[i:i + m]
        
        # Вычисляем расстояние Эвклида
        dist_profile[i] = ED_distance(query, ts_i_m)

    return dist_profile
