import numpy as np


def to_numpy_array(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, 'detach'):
        value = value.detach()

    if hasattr(value, 'cpu'):
        value = value.cpu()

    if hasattr(value, 'numpy'):
        return value.numpy()

    return np.asarray(value)
