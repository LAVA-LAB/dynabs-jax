import numpy as np
import jax.numpy as jnp
import jax
import time
import cdd
import itertools
from tqdm import tqdm
from .utils import create_batches

def forward_reach(step_set, state_min, state_max, input):
    vertices = step_set(state_min, state_max, input, input)

    return vertices

class RectangularForward(object):

    def __init__(self, regions, model):
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_forward_reach = jax.vmap(forward_reach, in_axes=(None, None, None, 0), out_axes=0)
        
        discrete_per_dimension = [np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i]) for i in range(len(model.num_actions))]
        discrete_inputs = np.array(list(itertools.product(*discrete_per_dimension)))

        t = time.time()

        vertices = {}
        pbar = tqdm(enumerate(zip(regions['lower_bounds'],regions['upper_bounds'])), total=len(regions['lower_bounds']))
        for i,(lb,ub) in pbar:
            # For every state, compute for every action the [lb,ub] forward reachable set
            vertices[i] = vmap_forward_reach(model.step_set, lb, ub, discrete_inputs)

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.inputs = discrete_inputs
        self.idxs = np.arange(len(discrete_inputs))
        self.vertices = vertices

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return