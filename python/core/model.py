import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time


def parse_model(base_model):
    '''
    Parse linear dynamical model
    '''

    print('Parse linear dynamical model...')
    t = time.time()

    base_model.partition['boundary'] = np.array(base_model.partition['boundary']).astype(float)
    base_model.partition['number_per_dim'] = np.array(base_model.partition['number_per_dim']).astype(int)

    # Control limitations
    base_model.uMin = np.array(base_model.uMin).astype(float)
    base_model.uMax = np.array(base_model.uMax).astype(float)

    lump = base_model.lump

    base_model.n = base_model.A.shape[0]

    if lump == 0:
        model = make_fully_actuated(base_model,
                                    manualDimension='auto')
    else:
        model = make_fully_actuated(base_model,
                                    manualDimension=lump)

    # Determine vertices of the control input space
    stacked = np.vstack((model.uMin, model.uMax))
    model.uVertices = np.array(list(itertools.product(*stacked.T)))

    # Determine inverse A matrix
    model.A_inv = np.linalg.inv(model.A)

    # Determine pseudo-inverse B matrix
    model.B_pinv = np.linalg.pinv(model.B)

    # Retreive system dimensions
    model.p = np.size(model.B, 1)  # Nr of inputs

    # Determine what the equilibrium point of the linear system is
    uAvg = (model.uMin + model.uMax) / 2
    if np.linalg.matrix_rank(np.eye(model.n) - model.A) == model.n:
        model.equilibrium = np.linalg.inv(np.eye(model.n) - model.A) @ \
                            (model.B @ uAvg + model.q)

    print(f'- Model parsing done (took {(time.time() - t):.3f} sec.)')
    print('')
    return model


def make_fully_actuated(model, manualDimension='auto'):
    '''
    Given a model in `model`, render it fully actuated.

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    manualDimension : int or str, optional
        Desired dimension of the state of the model The default is 'auto'.

    Returns
    -------
    model : dict
        Main dictionary of the LTI system model, which is now fully actuated.

    '''

    if manualDimension == 'auto':
        # Determine dimension for actuation transformation
        dim = int(np.size(model.A, 1) / np.size(model.B, 1))
    else:
        # Group a manual number of time steps
        dim = int(manualDimension)

    # Determine fully actuated system matrices and parameters
    A_hat = np.linalg.matrix_power(model.A, (dim))
    B_hat = np.concatenate([np.linalg.matrix_power(model.A, (dim - i)) \
                            @ model.B for i in range(1, dim + 1)], 1)

    q_hat = sum([np.linalg.matrix_power(model.A, (dim - i)) @ model.q
                 for i in range(1, dim + 1)])

    w_sigma_hat = sum([np.array(np.linalg.matrix_power(model.A, (dim - i))
                                @ model.noise['w_cov'] @
                                np.linalg.matrix_power(model.A.T, (dim - i))
                                ) for i in range(1, dim + 1)])

    # Overwrite original system matrices
    model.A = A_hat
    model.B = B_hat
    model.q = q_hat

    model.noise['w_cov'] = w_sigma_hat

    # Redefine sampling time of model
    model.tau *= dim

    model.uMin = np.repeat(model.uMin, dim)
    model.uMax = np.repeat(model.uMax, dim)

    return model
