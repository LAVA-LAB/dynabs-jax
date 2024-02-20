import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import itertools
import time
import os
import cdd
from tqdm import tqdm

from .scenario.create_table import create_table
from .scenario.load_table import load_table

from .utils import create_batches
from .polytope import num_points_in_polytope


def compute_scenario_interval_table(filepath, num_samples, confidence_level):

    if not os.path.isfile(filepath):

        print('The following table file does not exist:' + str(filepath))
        print('Create table now instead...')

        P_low, P_upp = create_table(N=num_samples, beta=confidence_level, kstep=1, trials=0,
                                    export=True, filepath=filepath)

        table = np.column_stack((P_low, P_upp))

    else:
        print('Loading scenario approach table...')

        # Load scenario approach table
        table = load_table(tableFile=filepath, k=num_samples)

    return table


def count_samples_per_region(args, model, partition, target_points, noise_samples, mode, batch_size=1000):

    print('Compute number of successor state samples in each partition element...')
    t = time.time()

    if args.debug:
        print('- Debug mode enabled (compare all methods)')

    result = {}
    if not partition.rectangular or args.debug:
        print('- Mode for nonrectangular partition')
        i = 0
        result[0] = count_general(partition, target_points, noise_samples, mode, batch_size)

    if partition.rectangular or args.debug:
        print('- Mode for rectangular partition')
        i = 1
        result[1] = count_rectangular(model, partition, target_points, noise_samples, batch_size)

    if args.debug:
        assert np.all(result[0] == result[1])

    print(f'Computing number of contained samples took {(time.time() - t):.3f} sec.')
    print('')
    return result[i]


vmap_num_points_in_polytope = jax.jit(jax.vmap(num_points_in_polytope, in_axes=(0, 0, None), out_axes=0))

@jax.jit
def compute_contained_for_single_action(d, noise_samples, As, bs):
    succ_samples = d + noise_samples
    num_samples_per_region = vmap_num_points_in_polytope(As, bs, succ_samples)

    return num_samples_per_region


vmap_compute_contained_for_single_action = jax.jit(jax.vmap(compute_contained_for_single_action,
                                                            in_axes=(0, None, None, None), out_axes=0))


def count_general(partition, target_points, noise_samples, mode, batch_size):

    @jax.jit
    def loop_body(i, val):
        As, bs, d, noise_samples, out = val

        out_curr = compute_contained_for_single_action(d[i], noise_samples, As, bs)

        out = out.at[i].set(out_curr)
        return (As, bs, d, noise_samples, out)

    if mode == 'fori_loop':

        num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)
        val = (partition.regions['A'], partition.regions['b'], target_points,
               noise_samples, num_samples_per_region)
        val = jax.lax.fori_loop(0, len(target_points), loop_body, val)
        (_, _, _, _, num_samples_per_region) = val

    else:
        # Use either vmap (if batch size > 1) or plain Python for loop

        if batch_size > 1:
            starts, ends = create_batches(len(target_points), batch_size)
            num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)

            for (i, j) in tqdm(zip(starts, ends), total=len(starts)):
                num_samples_per_region[i:j] = vmap_compute_contained_for_single_action(target_points[i:j],
                                                                                       noise_samples,
                                                                                       partition.regions['A'],
                                                                                       partition.regions['b'])

        else:
            num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)
            for i, d in tqdm(enumerate(target_points), total=len(target_points)):
                # Check if this action is enabled anywhere
                # if jnp.sum(enabled_actions[:, i]) > 0:
                num_samples_per_region[i] = compute_contained_for_single_action(d, noise_samples,
                                                                                partition.regions['A'],
                                                                                partition.regions['b'])

    return num_samples_per_region


def normalize_and_count(d, num_regions, noise_samples, lb, ub, number_per_dim, region_idx_array):
    '''
    Normalize the given samples, such that each region is a unit hypercube
    :param num_regions:
    :param d:
    :param noise_samples:
    :param lb:
    :param ub:
    :param number_per_dim:
    :param region_idx_array:
    :return:
    '''

    # Determine successor state samples
    samples = d + noise_samples

    # Discard samples outside of partition
    in_partition = jnp.all((samples >= lb) * (samples <= ub), axis=1)

    # Normalize samples
    samples_norm = (samples - lb) / (ub - lb) * number_per_dim

    # Perform integer division by 1 and determine to which regions the samples belong
    samples_idxs = jnp.array(samples_norm // 1, dtype=int)

    # If the integer division is below zero, the sample is outside the partition, but we want to avoid wrapping.
    # Thus, we set the index for these samples to zero, and increment all others by one
    samples_region_idxs = in_partition * (region_idx_array[tuple(samples_idxs.T)] + 1)

    # Determine counts for each index
    counts = jnp.bincount(samples_region_idxs, length=num_regions + 1)[1:]

    return counts


def count_rectangular(model, partition, target_points, noise_samples, batch_size):

    # If batch size is > 1, then use vmap version. Otherwise, use plain Python for loop.
    if batch_size > 1:
        # Define vmap function
        fn_vmap = jax.jit(jax.vmap(normalize_and_count, in_axes=(0, None, None, None, None, None, None), out_axes=0),
                          static_argnums=(1))

        starts, ends = create_batches(len(target_points), batch_size)
        num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)

        for (i, j) in tqdm(zip(starts, ends), total=len(starts)):
            num_samples_per_region[i:j] = fn_vmap(target_points[i:j],
                                             len(partition.regions['idxs']),
                                             noise_samples,
                                             model.partition['boundary'][0],
                                             model.partition['boundary'][1],
                                             model.partition['number_per_dim'],
                                             partition.region_idx_array)

    else:
        # Define jitted function
        fn = jax.jit(normalize_and_count, static_argnums=(1))

        num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)
        for i, d in tqdm(enumerate(target_points), total=len(target_points)):
            num_samples_per_region[i] = fn(d = d,
                                           num_regions = len(partition.regions['idxs']),
                                           noise_samples = noise_samples,
                                           lb = model.partition['boundary'][0],
                                           ub = model.partition['boundary'][1],
                                           number_per_dim = model.partition['number_per_dim'],
                                           region_idx_array = partition.region_idx_array)
        print('-- Number of times function was compiled:', fn._cache_size())

    return num_samples_per_region


def samples_to_intervals(num_samples, num_samples_per_region, interval_table, goal_bool, critical_bool):

    print('Convert number of contained samples to probability intervals...')
    t = time.time()

    # Sum over each row
    # num_samples_goal = np.sum(num_samples_per_region[:, goal_bool], axis=1)
    # num_samples_critical = np.sum(num_samples_per_region[:, critical_bool], axis=1)

    # print('Samples per region shape:', num_samples_per_region.shape)
    # print('Samples per region sum:', np.sum(num_samples_per_region, axis=1))

    print('- Determine number of samples in absorbing state...')
    num_samples_absorbing = num_samples - np.sum(num_samples_per_region, axis=1)

    # Exclude critical regions and goal regions
    # mask = ~goal_bool * ~critical_bool
    # num_samples_per_region_masked = num_samples_per_region[:, mask]

    # Read the probability intervals from the table (for the given number of samples per region)
    # P_masked = interval_table[num_samples - num_samples_per_region_masked]
    # P_full = np.zeros(num_samples_per_region.shape + (2,))
    # P_full[:, mask] = P_masked

    # P_goal = interval_table[num_samples - num_samples_goal]
    # P_critical = interval_table[num_samples - num_samples_critical]

    print('- Determine transition probability interval matrices...')
    P_absorbing = interval_table[num_samples - num_samples_absorbing]
    P_full = interval_table[num_samples - num_samples_per_region]

    # decmin = 6
    # pmin = 10**-decmin
    # print(f'- Put minimum (nonzero) probability to {pmin}')
    # P_full = np.maximum(pmin, np.round(P_full, decmin))
    # P_absorbing = np.maximum(pmin, np.round(P_absorbing, decmin))

    # If the sample count was zero, force probability to zero
    P_full[num_samples_per_region == 0] = 0

    print('- Perform checks...')
    # Perform checks on the transition probability intervals
    assert len(P_full) == len(P_absorbing)
    assert 0 <= np.all(P_full) <= 1
    assert 0 <= np.all(P_absorbing) <= 1
    # Check if all lower bounds sum up to <= 1 and upper bounds to >= 1
    assert np.all(np.sum(P_full[:,:,0], axis=1) + P_absorbing[:,0]) <= 1
    assert np.all(np.sum(P_full[:, :, 1], axis=1) + P_absorbing[:, 1]) >= 1

    print(f'Computing probability intervals took {(time.time() - t):.3f} sec.')
    print('')
    # return P_full, P_goal, P_critical, P_absorbing
    return P_full, P_absorbing

def sample_noise(model, key, number_samples):
    # Split noise key
    key, subkey = jax.random.split(key)

    # Compute Gaussian noise samples
    noise_samples = jax.random.multivariate_normal(key, np.zeros(model.n), model.noise['w_cov'],
                                                   shape=(number_samples,))

    return noise_samples
