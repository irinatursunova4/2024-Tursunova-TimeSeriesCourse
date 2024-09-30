import numpy as np
from metrics import ED_distance, norm_ED_distance, DTW_distance
from utils import z_normalize

class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize

    @property
    def distance_metric(self) -> str:
        """Return the distance metric"""
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """Choose distance function for calculation of matrix"""
        if self.metric == 'euclidean':
            return norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError("Unsupported metric: choose 'euclidean' or 'dtw'.")

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """Calculate distance matrix"""
        if np.any(np.isnan(input_data)):
            raise ValueError("Input data contains NaN values.")

        # Нормализация временных рядов, если необходимо
        if self.is_normalize:
            input_data = z_normalize(input_data)

        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        dist_func = self._choose_distance()

        for i in range(input_data.shape[0]):
            for j in range(i + 1, input_data.shape[0]):
                dist = dist_func(input_data[i], input_data[j])
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist  # матрица симметрична

        return matrix_values