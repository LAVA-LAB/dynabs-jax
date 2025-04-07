import itertools
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


@partial(jax.jit, static_argnums=(0))
def forward_reach(step_set, state_min, state_max, input, cov_diag, number_per_dim, cell_width, boundary_lb, boundary_ub):
    frs_min, frs_max = step_set(state_min, state_max, input, input)

    # If covariance is zero, then the span equals the number of cells the forward reachable set contains at most
    frs_span = jnp.astype(jnp.ceil((frs_max - frs_min) / cell_width), int)

    state_min_norm = (frs_min - boundary_lb) / (boundary_ub - boundary_lb) * number_per_dim
    lb_contained_in = state_min_norm // 1

    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) * (cov_diag == 0)).astype(int)
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) * (cov_diag == 0) + (number_per_dim - 1) * (cov_diag != 0)).astype(int)

    return frs_min, frs_max, frs_span, idx_low, idx_upp


class RectangularForward(object):

    def __init__(self, partition, model):
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_forward_reach = jax.vmap(forward_reach, in_axes=(None, None, None, 0, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0,))

        discrete_per_dimension = [np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i]) for i in range(len(model.num_actions))]
        discrete_inputs = np.array(list(itertools.product(*discrete_per_dimension)))

        t = time.time()

        frs = {}
        pbar = tqdm(enumerate(zip(partition.regions['lower_bounds'], partition.regions['upper_bounds'])), total=len(partition.regions['lower_bounds']))
        self.max_slice = np.zeros(model.n)
        for i, (lb, ub) in pbar:
            # For every state, compute for every action the [lb,ub] forward reachable set
            flb, fub, fsp, fil, fiu = vmap_forward_reach(model.step_set, lb, ub, discrete_inputs, model.noise['cov_diag'], partition.number_per_dim, partition.cell_width,
                                                         partition.boundary_lb, partition.boundary_ub)

            frs[i] = {}
            frs[i]['lb'] = flb
            frs[i]['ub'] = fub
            frs[i]['idx_lb'] = fil
            frs[i]['idx_ub'] = fiu

            self.max_slice = np.maximum(self.max_slice, jnp.max(fiu + 1 - fil, axis=0))
        self.max_slice = tuple(np.astype(self.max_slice, int).tolist())

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.inputs = discrete_inputs
        self.idxs = np.arange(len(discrete_inputs))
        self.frs = frs

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return
