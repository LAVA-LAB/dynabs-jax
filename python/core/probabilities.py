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
    '''
    Count the number of noise samples per abstract state (partition element).

    :param args:
    :param model:
    :param partition:
    :param target_points:
    :param noise_samples:
    :param mode:
    :param batch_size:
    :return: num_samples_per_state - integer-valued matrix, with each row an action and each column a (successor) state
    '''

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


def normalize_and_count(d, num_regions, noise_samples, lb, ub, number_per_dim, wrap, region_idx_array):
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
    samples_rounded = samples_norm // 1
    samples_idxs = jnp.array(samples_rounded * ~wrap + samples_rounded % number_per_dim * wrap, dtype=int)

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
        fn_vmap = jax.jit(jax.vmap(normalize_and_count, in_axes=(0, None, None, None, None, None, None, None), out_axes=0),
                          static_argnums=(1,2,3,4,5,6,7))

        starts, ends = create_batches(len(target_points), batch_size)
        num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)

        for (i, j) in tqdm(zip(starts, ends), total=len(starts)):
            num_samples_per_region[i:j] = fn_vmap(target_points[i:j],
                                                  len(partition.regions['idxs']),
                                                  noise_samples,
                                                  model.partition['boundary'][0],
                                                  model.partition['boundary'][1],
                                                  model.partition['number_per_dim'],
                                                  model.wrap,
                                                  partition.region_idx_array)

        print('-- Number of times function was compiled:', fn_vmap._cache_size())

    else:
        # Define jitted function
        fn = jax.jit(normalize_and_count) #, static_argnums=(1))

        num_samples_per_region = np.zeros((len(target_points), len(partition.regions['idxs'])), dtype=int)
        for i, d in tqdm(enumerate(target_points), total=len(target_points)):
            num_samples_per_region[i] = fn(d=d,
                                           num_regions=len(partition.regions['idxs']),
                                           noise_samples=noise_samples,
                                           lb=model.partition['boundary'][0],
                                           ub=model.partition['boundary'][1],
                                           number_per_dim=model.partition['number_per_dim'],
                                           wrap=model.wrap,
                                           region_idx_array=partition.region_idx_array)
        print('-- Number of times function was compiled:', fn._cache_size())

    return num_samples_per_region


def samples_to_intervals(num_samples, num_samples_per_region, interval_table, round_probabilities=False):
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

    if round_probabilities:
        decmin = 4
        pmin = 10 ** -decmin
        print(f'- Put minimum (nonzero) probability to {pmin}')
        P_full = np.maximum(pmin, np.round(P_full, decmin))
        P_absorbing = np.maximum(pmin, np.round(P_absorbing, decmin))

    # If the sample count was zero, force probability interval to zero
    P_full[num_samples_per_region == 0] = 0

    print('- Perform checks...')
    # Perform checks on the transition probability intervals
    assert len(P_full) == len(P_absorbing)
    # All probabilities are between 0 and 1
    assert np.all(0 <= P_full) and np.all(P_full <= 1)
    assert np.all(0 <= P_absorbing) and np.all(P_absorbing <= 1)
    # Check if all lower bounds sum up to <= 1 and upper bounds to >= 1
    assert np.all(np.sum(P_full[:, :, 0], axis=1) + P_absorbing[:, 0]) <= 1
    assert np.all(np.sum(P_full[:, :, 1], axis=1) + P_absorbing[:, 1]) >= 1

    # P_full[:,:,0] = num_samples_per_region / num_samples
    # P_full[:,:,1] = num_samples_per_region / num_samples
    # P_absorbing[:, 0] = num_samples_absorbing / num_samples
    # P_absorbing[:, 1] = num_samples_absorbing / num_samples

    print(f'Computing probability intervals took {(time.time() - t):.3f} sec.')
    print('')
    return P_full, P_absorbing


def sample_noise(model, key, number_samples):
    # Split noise key
    key, subkey = jax.random.split(key)

    # Compute Gaussian noise samples
    # noise_samples = jax.random.multivariate_normal(key, np.zeros(model.n), model.noise['w_cov'],
    #                                                shape=(number_samples,))

    noise_samples = np.random.multivariate_normal(np.zeros(model.n), model.noise['w_cov'],
                                  size=(number_samples,))

    return noise_samples


def count_single_box(lb_idx, ub_idx, idx_inv, number_per_dim, wrap):

    # Make sure to wrap specified variables correctly, which is done by shifting the region index rather than shifting the sample box
    shift_by = jnp.array(wrap * ((idx_inv - lb_idx) // number_per_dim) * number_per_dim, dtype=int)
    idx_inv += shift_by

    # Compute partition elements that intersect with the given box
    in_region = jnp.all(idx_inv >= lb_idx, axis=1) * jnp.all(idx_inv <= ub_idx, axis=1)

    # Compute if the given box is contained in a single partition element
    in_region_only = jnp.all(idx_inv == lb_idx, axis=1) * jnp.all(idx_inv == ub_idx, axis=1)

    # A box intersect with the complement of the partitioned space if any lower bound index is below 0 or upper bound above the number_per_dim
    in_absorbing = jnp.any((lb_idx < 0) * ~wrap) + jnp.any((ub_idx > number_per_dim) * ~wrap)

    # A box is completely outside the partitioned space if any upper bound index is below 0 or lower bound above the number_per_dim
    in_absorbing_only = jnp.any((ub_idx < 0) * ~wrap) + jnp.any((lb_idx > number_per_dim) * ~wrap)

    return in_region, in_region_only, in_absorbing, in_absorbing_only


# Vmap over multiple noise samples
vmap_count_single_box = jax.jit(jax.vmap(count_single_box, in_axes=(0, 0, None, None, None), out_axes=(0, 0, 0, 0)))


def normalize_and_count_box(d_lb, d_ub, noise_samples, lb, ub, number_per_dim, wrap, region_idx_inv):
    '''
    Normalize the forward reachable set for a *single action*, such that each region is a unit hypercube
    :param d:
    :param noise_samples:
    :param lb:
    :param ub:
    :param number_per_dim:
    :param wrap:
    :param region_idx_inv:
    :return:
    '''

    # Determine successor state samples
    lb_samples = d_lb + noise_samples
    ub_samples = d_ub + noise_samples

    # Normalize samples
    lb_samples_norm = (lb_samples - lb) / (ub - lb) * number_per_dim
    ub_samples_norm = (ub_samples - lb) / (ub - lb) * number_per_dim

    # Perform integer division by 1 and determine to which regions the samples belong
    lb_samples_idxs = jnp.array(lb_samples_norm // 1, dtype=int)
    ub_samples_idxs = jnp.array(ub_samples_norm // 1, dtype=int)

    in_region, in_region_only, in_absorbing, in_absorbing_only = vmap_count_single_box(lb_samples_idxs, ub_samples_idxs, region_idx_inv, number_per_dim, wrap)

    # Total counts are given by summing over the individual boxes
    region_lb = jnp.sum(in_region_only, axis=0)
    region_ub = jnp.sum(in_region, axis=0)
    absorbing_lb = jnp.sum(in_absorbing_only, axis=0)
    absorbing_ub = jnp.sum(in_absorbing, axis=0)

    return region_lb, region_ub, absorbing_lb, absorbing_ub


def count_rectangular_single_state(model, partition, reach_lb, reach_ub, noise_samples, batch_size):
    '''
    For a given state, compute the sample counts for all actions and all noise samples
    :param modeL:
    :param partition:
    :param reach_lb:
    :param reach_ub:
    :param noise_samples:
    :param batch_size:
    :return:
    '''

    region_lb = np.zeros((len(reach_lb), len(partition.regions['idxs'])), dtype=int)
    region_ub = np.zeros((len(reach_lb), len(partition.regions['idxs'])), dtype=int)
    absorbing_lb = np.zeros(len(reach_lb), dtype=int)
    absorbing_ub = np.zeros(len(reach_lb), dtype=int)

    # If batch size is > 1, then use vmap version. Otherwise, use plain Python for loop.
    if batch_size > 1:

        # Vmap over multiple actions
        fn_vmap = jax.jit(jax.vmap(normalize_and_count_box, in_axes=(0, 0, None, None, None, None, None, None), out_axes=(0, 0, 0, 0)))

        starts, ends = create_batches(len(reach_lb), batch_size)

        for (i, j) in tqdm(zip(starts, ends), total=len(starts)):
            result = fn_vmap(reach_lb[i:j],
                              reach_ub[i:j],
                              noise_samples,
                              model.partition['boundary'][0],
                              model.partition['boundary'][1],
                              model.partition['number_per_dim'],
                              model.wrap,
                              partition.region_idx_inv)

        region_lb[i:j] = result[0]
        region_ub[i:j] = result[1]
        absorbing_lb[i:j] = result[2]
        absorbing_ub[i:j] = result[3]

        print('-- Number of times function was compiled:', fn_vmap._cache_size())

    else:
        # Define jitted function
        fn = jax.jit(normalize_and_count_box) #, static_argnums=(2,3,4,5,6,7))

        for i, (d_lb, d_ub) in tqdm(enumerate(zip(reach_lb, reach_ub)), total=len(reach_lb)):

            result = fn(
                d_lb,
                d_ub,
                noise_samples,
                model.partition['boundary'][0],
                model.partition['boundary'][1],
                model.partition['number_per_dim'],
                model.wrap,
                partition.region_idx_inv)

            region_lb[i] = result[0]
            region_ub[i] = result[1]
            absorbing_lb[i] = result[2]
            absorbing_ub[i] = result[3]

        print('-- Number of times function was compiled:', fn._cache_size())

    return region_lb, region_ub, absorbing_lb, absorbing_ub