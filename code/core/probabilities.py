import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
import cdd
from tqdm import tqdm

from .polytope import num_points_in_polytope
vmap_num_points_in_polytope = jax.jit(jax.vmap(num_points_in_polytope, in_axes=(0, 0, None), out_axes=0))

@jax.jit
def compute_contained_for_single_action(samples, As, bs):

    num_samples_per_region = vmap_num_points_in_polytope(As, bs, samples)

    return num_samples_per_region

def compute_num_contained_all_actions(partition, actions, enabled_actions, noise_samples, mode):
    print('Compute transition probability intervals...')
    t = time.time()

    @jax.jit
    def loop_body(i, val):
        As, bs, d, noise_samples, out = val

        succ_samples = d[i] + noise_samples
        out_curr = compute_contained_for_single_action(succ_samples, As, bs)

        out = out.at[i].set(out_curr)
        return (As, bs, d, noise_samples, out)

    if mode == 'fori_loop':

        num_samples_per_region = np.zeros((len(actions.backreach['target_points']), len(partition.regions['idxs'])))
        val = (partition.regions['A'], partition.regions['b'], actions.backreach['target_points'],
               noise_samples, num_samples_per_region)
        val = jax.lax.fori_loop(0, len(actions.backreach['target_points']), loop_body, val)
        (_, _, _, _, num_samples_per_region) = val

    else:

        num_samples_per_region = np.zeros((len(actions.backreach['target_points']), len(partition.regions['idxs'])))

        for i, d in tqdm(enumerate(actions.backreach['target_points'])):
            # Check if this action is enabled anywhere
            if jnp.sum(enabled_actions[:, i]) > 0:
                succ_samples = d + noise_samples

                num_samples_per_region[i] = compute_contained_for_single_action(succ_samples, partition.regions['A'],
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