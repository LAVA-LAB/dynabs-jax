import numpy as np
import jax.numpy as jnp
import jax

from benchmarks.Drone2D import Drone2D
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions

# Test switch
TEST = True

# Define and parse model
base_model = Drone2D()
model = parse_model(base_model)

mode = ['fori_loop', 'vmap', 'python'][0]
key = jax.random.PRNGKey(0)

partition = RectangularPartition(number_per_dim=model.partition['number_per_dim'],
                                 partition_boundary=model.partition['boundary'],
                                 goal_regions=model.goal,
                                 critical_regions=model.critical,
                                 mode = mode)

actions = RectangularTarget(target_points=partition.regions['centers'],
                            model=model)
actions.test_backwardset(idx=10, model=model)

enabled_actions = compute_enabled_actions(jnp.array(actions.backreach['A']),
                                          jnp.array(actions.backreach['b']),
                                          jnp.array(partition.regions['all_vertices']),
                                          mode = mode)
print('Total number of enabled actions:', np.sum(np.any(enabled_actions, axis=0)))

# %%

from core.probabilities import sample_noise, compute_num_contained_all_actions

NUM_SAMPLES = 10000
samples = sample_noise(model, key, NUM_SAMPLES)

num_samples_per_region3 = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
                                                            samples, mode = 'vmap', batch_size = 4)

num_samples_per_region = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
                                                           samples, mode = 'fori_loop')

if TEST:
    num_samples_per_region2 = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
                                                                samples, mode = 'python')

    assert np.all(num_samples_per_region == num_samples_per_region2)