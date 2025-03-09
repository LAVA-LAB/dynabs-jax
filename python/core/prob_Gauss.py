import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial

# Note: The following implementation only works for Gaussian distributions with diagonal covariance

@jax.jit
def integ_Gauss(x_lb, x_ub, x, cov):
    eps = 1e-9 # Add tiny epsilon to avoid NaN problems if the Gaussian is a Dirac (i.e., cov=0) and x_lb or x_ub equals x
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

    for i in range(n):
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

    return probs

@jax.jit
def interval_distribution(x_lbs, x_ubs, mean_lb, mean_ub, cov, state_space_lb, state_space_ub):

    prob = vmap_minmax_Gauss(x_lbs, x_ubs, mean_lb, mean_ub, cov)
    prob_nonzero = prob[:,1] > 1e-6

    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov)
    prob_absorbing = 1 - prob_state_space[::-1]

    return prob, prob_nonzero, prob_absorbing

# vmap to compute distributions for all actions in a state
vmap_interval_distribution = jax.jit(jax.vmap(interval_distribution, in_axes=(None, None, 0, 0, None, None, None), out_axes=(0, 0, 0)))

@partial(jax.jit, static_argnums=(0))
def interval_distribution_per_dim(n, x_lb_per_dim, x_ub_per_dim, region_idx_inv, mean_lb, mean_ub, cov, state_space_lb, state_space_ub):

    # Compute the probability intervals for each dimension
    probs = minmax_Gauss_per_dim(n, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, cov)

    # Stack the probability intervals over all dimensions
    # prob_stack = jnp.stack([probs[i][region_idx_inv[:,i]] for i in range(n)], axis=2)
    #
    # # Take the product of the lower bounds and the upper bounds to obtain the overall probability intervals
    # prob = jnp.prod(prob_stack, axis=2)
    prob = probs[0][region_idx_inv[:,0]]
    for i in range(1,n):
        prob = jnp.multiply(prob, probs[i][region_idx_inv[:,i]])
    prob_nonzero = prob[:,1] > 1e-6

    # Compute probability to end outside of partition
    # TODO: Account for wrapping variables
    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov)
    prob_absorbing = 1 - prob_state_space[::-1]

    return prob, prob_nonzero, prob_absorbing

# vmap to compute distributions for all actions in a state
vmap_interval_distribution_per_dim = jax.jit(jax.vmap(interval_distribution_per_dim, in_axes=(None, None, None, None, 0, 0, None, None, None), out_axes=(0, 0, 0)), static_argnums=(0,))

def compute_probabilities(model, partition, reach):

    prob = {}
    prob_idx = {}
    prob_absorbing = {}
    states = np.arange(partition.size)

    # For all states
    for s, reach_state in tqdm(enumerate(reach.values()), total=len(reach)):

        prob[s] = {}
        prob_idx[s] = {}
        prob_absorbing[s] = {}

        # Compute the probability distribution for every action
        p, p_nonzero, pa = vmap_interval_distribution(partition.regions['lower_bounds'], partition.regions['upper_bounds'],
                                   reach_state[0], reach_state[1], model.noise['cov'],
                                   partition.boundary_lb, partition.boundary_ub)

        p = np.array(p)
        p_nonzero = np.array(p_nonzero)
        pa = np.array(pa)

        for a in range(len(reach_state[0])):
            prob[s][a] = p[a][p_nonzero[a]]
            prob_idx[s][a] = states[p_nonzero[a]]
            prob_absorbing[s][a] = pa[a]

    return prob, prob_absorbing


def compute_probabilities_per_dim(model, partition, reach):

    prob = {}
    prob_idx = {}
    prob_absorbing = {}
    states = np.arange(partition.size)

    # For all states
    for s, reach_state in tqdm(enumerate(reach.values()), total=len(reach)):

        prob[s] = {}
        prob_idx[s] = {}
        prob_absorbing[s] = {}

        # Compute the probability distribution for every action
        p, p_nonzero, pa = vmap_interval_distribution_per_dim(model.n,
                                                              partition.regions_per_dim['lower_bounds'],
                                                              partition.regions_per_dim['upper_bounds'],
                                                              partition.region_idx_inv,
                                                              reach_state[0],
                                                              reach_state[1],
                                                              model.noise['cov'],
                                                              partition.boundary_lb,
                                                              partition.boundary_ub)

        p = np.array(p)
        p_nonzero = np.array(p_nonzero)
        pa = np.array(pa)

        for a in range(len(reach_state[0])):
            prob[s][a] = p[a][p_nonzero[a]]
            prob_idx[s][a] = states[p_nonzero[a]]
            prob_absorbing[s][a] = pa[a]

    return prob, prob_absorbing