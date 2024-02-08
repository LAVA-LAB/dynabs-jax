import jax
import jax.numpy as jnp
import numpy as np

def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.
    :param data:
    :param batch_size:
    :return:
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = np.arange(num_batches) * batch_size
    ends = np.minimum(starts + batch_size, data_length)

    return starts, ends