import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Note: The following implementation only works for Gaussian distributions with diagonal covariance

@jax.jit
def integ_Gauss(x_lb, x_ub, x, cov):
    return jax.scipy.stats.norm.cdf(x_ub, x, cov) - jax.scipy.stats.norm.cdf(x_lb, x, cov)

# vmap to compute multivariate Gaussian integral
vmap_integ_Gauss = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, 0, 0), out_axes=0))

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

@jax.jit
def interval_distribution(x_lbs, x_ubs, mean_lb, mean_ub, cov, state_space_lb, state_space_ub):

    prob = vmap_minmax_Gauss(x_lbs, x_ubs, mean_lb, mean_ub, cov)

    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov)
    prob_absorbing = 1 - prob_state_space[::-1]

    return prob, prob_absorbing

# vmap to compute distributions for all actions in a state
vmap_interval_distribution = jax.jit(jax.vmap(interval_distribution, in_axes=(None, None, 0, 0, None, None, None), out_axes=(0, 0)))

def compute_probabilities(model, partition, reach):

    prob = {}
    prob_idx = {}
    prob_absorbing = {}

    # For all states
    for s, reach_state in tqdm(enumerate(reach.values()), total=len(reach)):

        prob[s] = {}
        prob_idx[s] = {}
        prob_absorbing[s] = {}

        # Compute the probability distribution for every action
        p, pa = vmap_interval_distribution(partition.regions['lower_bounds'], partition.regions['upper_bounds'],
                                   reach_state[0], reach_state[1], model.noise['cov'],
                                   partition.boundary_lb, partition.boundary_ub)

        for a in range(len(reach_state[0])):
            print(p[:,a,0] > 1e-6)
            prob[s][a] = p[p[:,a,0] > 1e-6]
            prob_idx[s][a] = np.arange(partition.size)
            prob_absorbing[s][a] = pa[a]

    return prob, prob_absorbing