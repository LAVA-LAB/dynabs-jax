import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial, reduce
import time
import itertools

# Note: The following implementation only works for Gaussian distributions with diagonal covariance

@partial(jax.jit, static_argnums=(2))
def dynslice(V, idx_low, size):
    roll = jnp.roll(V, -idx_low)
    # roll_zero = roll.at[size:].set(0)
    return roll[:size]

@jax.jit
def integ_Gauss(x_lb, x_ub, x, cov):
    eps = 1e-5 # Add tiny epsilon to avoid NaN problems if the Gaussian is a Dirac (i.e., cov=0) and x_lb or x_ub equals x
    return jax.scipy.stats.norm.cdf(x_ub, x+eps, cov) - jax.scipy.stats.norm.cdf(x_lb, x+eps, cov)

# vmap to compute multivariate Gaussian integral in n dimensions
vmap_integ_Gauss = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, 0, 0), out_axes=0))
vmap_integ_Gauss_per_dim = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, 0, None), out_axes=0))
vmap_integ_Gauss_per_dim_single = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, None, None), out_axes=0))

@jax.jit
def minmax_Gauss(x_lb, x_ub, mean_lb, mean_ub, cov):

    # Determine point closest to mean of region over which to integrate
    mean = (x_lb + x_ub) / 2
    closest_to_mean = jnp.maximum(jnp.minimum(mean_ub, mean), mean_lb)

    # Maximum probability is the product
    p_max = jnp.prod(vmap_integ_Gauss(x_lb, x_ub, closest_to_mean, jnp.diag(cov)))

    p1 = vmap_integ_Gauss(x_lb, x_ub, mean_lb, jnp.diag(cov))
    p2 = vmap_integ_Gauss(x_lb, x_ub, mean_ub, jnp.diag(cov))
    p_min = jnp.prod(jnp.minimum(p1, p2))

    return jnp.array([p_min, p_max])

# vmap to compute full distribution for one state-action pair
vmap_minmax_Gauss = jax.jit(jax.vmap(minmax_Gauss, in_axes=(0, 0, None, None, None), out_axes=0))

@partial(jax.jit, static_argnums=(0,))
def minmax_Gauss_per_dim(n, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, cov):
    '''
    Exploit rectangular partition to compute much fewer Gaussian integrals
    '''

    probs = [[] for _ in range(n)]
    prob_low = [[] for _ in range(n)]
    prob_high = [[] for _ in range(n)]

    for i in range(n):
        if n == 2:
            p_max = 0
            p_min = 0
            for shift in [-10,0,10]:
                x_lb = x_lb_per_dim[i] + shift
                x_ub = x_ub_per_dim[i] + shift

                # Determine point closest to mean of region over which to integrate
                mean = (x_lb + x_ub) / 2
                closest_to_mean = jnp.maximum(jnp.minimum(mean_ub[i], mean), mean_lb[i])

                # Maximum probability is the product
                p_max += vmap_integ_Gauss_per_dim(x_lb, x_ub, closest_to_mean, cov[i, i])

                p1 = vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_lb[i], cov[i, i])
                p2 = vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_ub[i], cov[i, i])
                p_min += jnp.minimum(p1, p2)

        else:
            x_lb = x_lb_per_dim[i]
            x_ub = x_ub_per_dim[i]

            # Determine point closest to mean of region over which to integrate
            mean = (x_lb + x_ub) / 2
            closest_to_mean = jnp.maximum(jnp.minimum(mean_ub[i], mean), mean_lb[i])

            # Maximum probability is the product
            p_max = vmap_integ_Gauss_per_dim(x_lb, x_ub, closest_to_mean, cov[i,i])

            p1 = vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_lb[i], cov[i,i])
            p2 = vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_ub[i], cov[i,i])
            p_min = jnp.minimum(p1, p2)

        probs[i] = jnp.vstack([p_min, p_max]).T
        prob_low[i] = p_min
        prob_high[i] = p_max

    return probs, prob_low, prob_high

@jax.jit
def interval_distribution(x_lbs, x_ubs, mean_lb, mean_ub, cov, state_space_lb, state_space_ub):

    prob = vmap_minmax_Gauss(x_lbs, x_ubs, mean_lb, mean_ub, cov)
    prob_nonzero = prob[:,1] > 1e-6

    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov)
    prob_absorbing = 1 - prob_state_space[::-1]

    return prob, prob_nonzero, prob_absorbing

# vmap to compute distributions for all actions in a state
vmap_interval_distribution = jax.jit(jax.vmap(interval_distribution, in_axes=(None, None, 0, 0, None, None, None), out_axes=(0, 0, 0)))

@partial(jax.jit, static_argnums=(0,1))
def interval_distribution_per_dim(n, max_slice, number_per_dim, per_dim_lb, per_dim_ub, i_lb, mean_lb, mean_ub, cov, state_space_lb, state_space_ub, region_idx_array):

    # Extract slices from the partition elements per dimension
    x_lb = [dynslice(per_dim_lb[i], i_lb[i], max_slice[i]) for i in range(n)]
    x_ub = [dynslice(per_dim_ub[i], i_lb[i], max_slice[i]) for i in range(n)]

    # List of indexes of the partition elements in the slices above
    prob_idx = [jnp.arange(max_slice[i]) + i_lb[i] for i in range(n)]

    # Compute the probability intervals for each dimension
    _, prob_low, prob_high = minmax_Gauss_per_dim(n, x_lb, x_ub, mean_lb, mean_ub, cov)

    prob_low_prod = reduce(jnp.multiply.outer, prob_low).flatten()
    prob_high_prod = reduce(jnp.multiply.outer, prob_high).flatten()

    # Note: meshgrid is used to get the Cartesian product between the indexes of the partition elements in every state space dimension, but meshgrid sorts in the wrong order.
    # To fix this, we first flip the order of the dimensions, then compute the meshgrid, and again flip the columns of the result. This ensures the sorting is in the correct order.
    prob_idx_flip = [prob_idx[n-i-1] for i in range(n)]
    prob_idx = jnp.flip(jnp.asarray(jnp.meshgrid(*prob_idx_flip, indexing='ij')).T.reshape(-1, n), axis=1)

    prob_idx_clip = jnp.astype(jnp.clip(prob_idx, jnp.zeros(n), number_per_dim), int)
    prob_id = region_idx_array[tuple(prob_idx_clip.T)]

    # Only keep nonzero probabilities, and also filter spurious indices that were added to keep arrays in JAX of fixed size
    prob_nonzero = (prob_high_prod > 1e-4) * jnp.all(prob_idx < number_per_dim, axis=1)

    # For the nonzero probabilities, also set a (very small) minimum lower bound probability (to ensure the IMDP is "graph-preserving")
    # Also round probabilities to a specified number of decimals
    decmin = 4
    pmin = 10 ** -decmin
    prob_low_prod = jnp.maximum(pmin * prob_nonzero, jnp.round(prob_low_prod, decmin))
    prob_high_prod = jnp.maximum(pmin * prob_nonzero, jnp.round(prob_high_prod, decmin))

    # Stack lower and upper bounds such that such prob[s] is an array of length two representing a single interval
    prob = jnp.stack([prob_low_prod, prob_high_prod]).T

    # Compute probability to end outside of partition
    # TODO: Account for wrapping variables
    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov)
    prob_absorbing = 1 - prob_state_space[::-1]

    if n != 2:
        prob_absorbing = jnp.maximum(pmin * (prob_absorbing[1] > 0), jnp.round(prob_absorbing, decmin))
    else:
        prob_absorbing = 0

    return prob, prob_idx, prob_id, prob_nonzero, prob_absorbing

# vmap to compute distributions for all actions in a state
vmap_interval_distribution_per_dim = jax.jit(jax.vmap(interval_distribution_per_dim, in_axes=(None, None, None, None, None, 0, 0, 0, None, None, None, None), out_axes=(0, 0, 0, 0, 0)), static_argnums=(0,1))

def compute_probabilities(model, partition, frs):

    prob = {}
    prob_idx = {}
    prob_absorbing = {}
    states = np.arange(partition.size)

    # For all states
    for s, frs_s in tqdm(enumerate(frs.values()), total=len(frs)):

        prob[s] = {}
        prob_idx[s] = {}
        prob_absorbing[s] = {}

        # Compute the probability distribution for every action
        p, p_nonzero, pa = vmap_interval_distribution(partition.regions['lower_bounds'], partition.regions['upper_bounds'],
                                   frs_s['lb'], frs_s['ub'], model.noise['cov'],
                                   partition.boundary_lb, partition.boundary_ub)

        p = np.array(p)
        p_nonzero = np.array(p_nonzero)
        pa = np.array(pa)

        for a in range(len(frs[0]['lb'])):
            prob[s][a] = p[a] #[p_nonzero[a]]
            prob_idx[s][a] = states[p_nonzero[a]]
            prob_absorbing[s][a] = pa[a]

    return prob, prob_absorbing


def compute_probabilities_per_dim(model, partition, frs, max_slice):

    prob = {}
    prob_idx = {}
    prob_id = {}
    prob_nonzero = {}
    prob_absorbing = {}
    # states = np.arange(partition.size)

    # For all states
    for s, frs_s in tqdm(enumerate(frs.values()), total=len(frs)):
        p, p_idx, p_id, p_nonzero, pa = vmap_interval_distribution_per_dim(model.n,
                                                              max_slice,
                                                              partition.number_per_dim,
                                                              partition.regions_per_dim['lower_bounds'],
                                                              partition.regions_per_dim['upper_bounds'],
                                                              frs_s['idx_lb'],
                                                              frs_s['lb'],
                                                              frs_s['ub'],
                                                              model.noise['cov'],
                                                              partition.boundary_lb,
                                                              partition.boundary_ub,
                                                              partition.region_idx_array)

        prob[s] = np.array(p)
        prob_idx[s] = np.array(p_idx)
        prob_id[s] = np.array(p_id)
        prob_nonzero[s] = np.array(p_nonzero)
        prob_absorbing[s] = np.array(pa)

        nans = np.where(np.any(np.isnan(prob[s]), axis=0))[0]
        if len(nans) > 0:
            print('NaN probabilities in state {} at position {}'.format(s, len(nans)))

    print('-- Number of times function was compiled:', interval_distribution_per_dim._cache_size())

    return prob, prob_idx, prob_id, prob_nonzero, prob_absorbing

@partial(jax.jit, static_argnums=(0))
def compose(n, prob_low, prob_high, nonzero_id):
    prob_low_outer = reduce(jnp.multiply.outer, prob_low).flatten()
    prob_high_outer = reduce(jnp.multiply.outer, prob_high).flatten()
    prob = jnp.stack([prob_low_outer, prob_high_outer]).T

    idx = jnp.asarray(jnp.meshgrid(*nonzero_id)).T.reshape(-1, n)

    return prob, idx