import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
    profile_distances = np.copy(matrix_profile['mp'])
    profile_indices = matrix_profile['mpi']
    
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    neg_inf_value = -np.inf

    for _ in range(top_k):
        discord_idx = np.argmax(profile_distances)
        discord_dist = profile_distances[discord_idx]
        nn_idx = profile_indices[discord_idx]

        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)

        profile_distances = apply_exclusion_zone(profile_distances, discord_idx, matrix_profile['excl_zone'],
                                                 neg_inf_value)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
