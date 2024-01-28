import numpy as np
import jax.numpy as jnp
import jax

from benchmarks.Drone2D import Drone2D
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions
from core.probabilities import sample_noise, compute_num_contained_all_actions

# Define and parse model
base_model = Drone2D()
model = parse_model(base_model)

mode = ['fori_loop', 'vmap', 'python'][0]
NUM_SAMPLES = 100
key = jax.random.PRNGKey(0)

partition = RectangularPartition(number_per_dim=model.partition['number_per_dim'],
                                 partition_boundary=model.partition['boundary'],
                                 goal_regions=model.goal,
                                 critical_regions=model.critical,
                                 mode = mode)

actions = RectangularTarget(target_points=partition.regions['centers'],
                            model=model)

enabled_actions = compute_enabled_actions(jnp.array(actions.backreach['A']),
                                          jnp.array(actions.backreach['b']),
                                          jnp.array(partition.regions['all_vertices']),
                                          mode = mode)

samples = sample_noise(model, key, NUM_SAMPLES)

num_samples_per_region = compute_num_contained_all_actions(partition, actions, enabled_actions, samples)