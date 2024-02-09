import numpy as np
import jax.numpy as jnp
import jax
import os
from pathlib import Path
from benchmarks.Drone2D import Drone2D
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions
from core.probabilities import sample_noise, compute_num_contained_all_actions, compute_scenario_interval_table, \
    samples_to_intervals
from core.imdp import Builder

cwd = os.path.dirname(os.path.abspath(__file__))
root_dir = Path(cwd)

# Test switch
TEST = True

# Define and parse model
base_model = Drone2D()
model = parse_model(base_model)

mode = ['fori_loop', 'vmap', 'python'][1]
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

NUM_SAMPLES = 3200
CONFIDENCE = 0.01
samples = sample_noise(model, key, NUM_SAMPLES)

# num_samples_per_region3 = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
#                                                             samples, mode = 'vmap', batch_size = 2)

# num_samples_per_region2 = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
#                                                             samples, mode = 'fori_loop')

num_samples_per_region = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
                                                           samples, mode = 'python')

table_filename = f'intervals_N={NUM_SAMPLES}_beta={CONFIDENCE}.csv'
interval_table = compute_scenario_interval_table(Path(str(root_dir), 'interval_tables', table_filename),
                                                 NUM_SAMPLES, CONFIDENCE)

# Compute probability intervals
P_full, P_absorbing = samples_to_intervals(NUM_SAMPLES,
                                                               num_samples_per_region,
                                                               interval_table,
                                                               partition.goal['bools'],
                                                               partition.critical['bools'])

# %%

import stormpy
from core.imdp import imdp_test2

matrix, imdp, result = imdp_test2()
stormpy.export_to_drn(imdp, 'out_test.drn')
print('Result:', result)

# %%

# Build interval MDP via StormPy
builder = Builder(states=partition.regions['idxs'],
                 goal_regions=partition.goal['idxs'],
                 critical_regions=partition.critical['idxs'],
                 actions=actions.backreach['idxs'],
                 enabled_actions=enabled_actions,
                 P_full=P_full,
                 P_absorbing=P_absorbing)

stormpy.export_to_drn(builder.imdp, 'out.drn')

builder.compute_reach_avoid()
i = 667
print(f'Result at state {partition.regions["centers"][i]} is: {builder.result_robust.at(i)}')

# %%

s = 786
a = 0

for transition in builder.imdp.states[s].actions[a].transitions:
    print("From state {} by action {}, with probability {}, go to state {}".format(s, a, transition.value(),
                                                                                   transition.column))
