import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
import cdd
from tqdm import tqdm

from .polytope import num_points_in_polytope

@jax.jit
def compute_intervals_single_action(samples, As, bs):

    vmap_num_points_in_polytope = jax.vmap(num_points_in_polytope, in_axes=(0, 0, None), out_axes=0)
    num_samples_per_region = vmap_num_points_in_polytope(As, bs, samples)

    return num_samples_per_region

def compute_num_contained_all_actions(partition, actions, enabled_actions, noise_samples):
    print('Compute transition probability intervals...')

    num_samples_per_region = np.zeros((len(actions.backreach['target_points']), len(partition.regions['idxs'])))

    t = time.time()

    for i,d in tqdm(enumerate(actions.backreach['target_points'])):
        # Check if this action is enabled anywhere
        if jnp.sum(enabled_actions[:,i]) > 0:

            succ_samples = d + noise_samples

            num_samples_per_region[i] = compute_intervals_single_action(succ_samples, partition.regions['A'],
                                                                        partition.regions['b'])

    print(f'- Number of samples for each transition computed (took {(time.time()-t):.3f} sec.)')

    return np.array(num_samples_per_region)

def sample_noise(model, key, number_samples):

    # Split noise key
    key, subkey = jax.random.split(key)

    # Compute Gaussian noise samples
    noise_samples = jax.random.multivariate_normal(key, np.zeros(model.n), model.noise['w_cov'],
                                                   shape=(number_samples,))

    return noise_samples