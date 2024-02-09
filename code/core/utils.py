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

def lexsort4d(array):

    idxs = np.lexsort((
        array[:,3],
        array[:,2],
        array[:,1],
        array[:,0]
    ))

    return array[idxs]