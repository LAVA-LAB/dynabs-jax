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
    starts = np.array(np.arange(num_batches) * batch_size, dtype=int)
    ends = np.array(np.minimum(starts + batch_size, data_length), dtype=int)

    return starts, ends

def lexsort4d(array):

    idxs = np.lexsort((
        array[:,3],
        array[:,2],
        array[:,1],
        array[:,0]
    ))

    return array[idxs]

def writeFile(file, operation="w", content=[""]):
    '''
    Create a filehandle and store content in it.

    Parameters
    ----------
    file : str
        Filename to store the content in.
    operation : str, optional
        Type of operation to perform on the file. The default is "w".
    content : list, optional
        List of strings to store in the file. The default is [""].

    Returns
    -------
    None.

    '''
    filehandle = open(file, operation)
    filehandle.writelines(content)
    filehandle.close()