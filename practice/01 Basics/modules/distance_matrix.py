import numpy as np

from modules.metrics import *
from modules.utils import z_normalize


class DistanceMatrix:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric : distance metric between two time series
    normalize : normalize or not time series
                Options: {euclidean, dtw}
    """

    def __init__(self, metric : str = 'euclidean', normalize : bool = False) -> None:

        self._metric : str = metric
        self._normalize : bool = normalize
        self._shape : tuple[int, int] = (0, 0) 
        self._values : np.ndarray | None = None


    @property
    def values(self) -> np.ndarray:
        """Return distances between time series
        
        Returns
        -------
        _values: distance matrix values
        """

        return self._values


    @property
    def shape(self) -> tuple[int, int]:
        """ Return the dimensionality of distance matrix
        
        Returns
        -------
        shape: size of distance matrix
        """

        return self._shape


    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            metric which is used to calculate distances between time series  
        """

        normalization_str = ""
        if (self._normalize):
            normalization_str = "normalized "
        else:
            normalization_str = "non-normalized "

        return normalization_str + self._metric + " distance" 


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dict_func: function reference
        """

        dist_func = None

       # INSERT YOUR CODE

        return dist_func


    def calculate(self, input_data: np.ndarray) -> None:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        """

        # INSERT YOUR CODE
