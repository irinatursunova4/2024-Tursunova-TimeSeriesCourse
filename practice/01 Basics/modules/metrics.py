import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two time series.

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    if len(ts1) != len(ts2):
        raise ValueError("Time series must have the same length")

    # Вычисление евклидова расстояния
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    return ed_dist

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)

    # Вычисляем средние и стандартные отклонения
    mean_ts1 = np.mean(ts1)
    mean_ts2 = np.mean(ts2)

    std_ts1 = np.std(ts1)
    std_ts2 = np.std(ts2)

    # Вычисляем скалярное произведение временных рядов
    dot_product = np.dot(ts1, ts2)

    # Нормализованное расстояние
    norm_ed_dist = np.sqrt(2 * n * (1 - dot_product / (n * std_ts1 * std_ts2)))

    return norm_ed_dist

def DTW_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Вычисление DTW расстояния между двумя временными рядами одинаковой длины.

    Параметры
    ----------
    ts1: np.ndarray
        Первый временной ряд.
    ts2: np.ndarray
        Второй временной ряд.

    Возвращает
    -------
    float
        DTW расстояние между ts1 и ts2.
    """

    n = len(ts1)

    # Инициализация матрицы DTW
    dtw = np.full((n + 1, n + 1), np.inf)
    dtw[0, 0] = 0  # Начальная позиция

    # Заполнение матрицы DTW
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2  # Евклидово расстояние
            dtw[i, j] = cost + min(dtw[i - 1, j],    # Вверх
                                   dtw[i, j - 1],    # Влево
                                   dtw[i - 1, j - 1]) # Вверх-влево

    return dtw[n, n]