import numpy as np

def swap(coords: np.ndarray) -> np.ndarray:
    """
    Returns a new numpy array where the x and y coordinates are swapped.
    """
    swapped = coords.copy()

    swapped[:, [0, 1, 2, 3]] = swapped[:, [1, 0, 3, 2]]

    return swapped
