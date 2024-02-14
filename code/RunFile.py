import numpy as np
import jax.numpy as jnp
import jax
import os
import time
from pathlib import Path
from benchmarks.Drone2D import Drone2D
from core.options import parse_arguments
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions
from core.probabilities import sample_noise, compute_num_contained_all_actions, compute_scenario_interval_table, \
    samples_to_intervals
from core.imdp import BuilderStorm, BuilderPrism

args = parse_arguments()
np.random.seed(args.seed)
args.jax_key = jax.random.PRNGKey(args.seed)

# Set current working directory
args.cwd = os.path.dirname(os.path.abspath(__file__))
args.root_dir = Path(args.cwd)

print('Run using arguments:')
for key,val in vars(args).items():
    print(' - `'+str(key)+'`: '+str(val))

# Define and parse model
base_model = Drone2D()
model = parse_model(base_model)

partition = RectangularPartition(number_per_dim=model.partition['number_per_dim'],
                                 partition_boundary=model.partition['boundary'],
                                 goal_regions=model.goal,
                                 critical_regions=model.critical,
                                 mode = args.mode)

actions = RectangularTarget(target_points=partition.regions['centers'],
                            model=model)
actions.test_backwardset(idx=10, model=model)

enabled_actions = compute_enabled_actions(jnp.array(actions.backreach['A']),
                                          jnp.array(actions.backreach['b']),
                                          jnp.array(partition.regions['all_vertices']),
                                          mode = args.mode)

samples = sample_noise(model, args.jax_key, args.num_samples)

num_samples_per_region = compute_num_contained_all_actions(partition, actions.backreach['target_points'],
                                                           samples, mode = args.mode, batch_size=100)

table_filename = f'intervals_N={args.num_samples}_beta={args.confidence}.csv'
interval_table = compute_scenario_interval_table(Path(str(args.root_dir), 'interval_tables', table_filename),
                                                 args.num_samples, args.confidence)

# Compute probability intervals
P_full, P_absorbing = samples_to_intervals(args.num_samples,
                                           num_samples_per_region,
                                           interval_table,
                                           partition.goal['bools'],
                                           partition.critical['bools'])

# %%

if args.checker == 'storm' or args.debug:

    # Build interval MDP via StormPy
    t = time.time()
    builderS = BuilderStorm(states=partition.regions['idxs'],
                            goal_regions=partition.goal['idxs'],
                            critical_regions=partition.critical['idxs'],
                            actions=actions.backreach['idxs'],
                            enabled_actions=enabled_actions,
                            P_full=P_full,
                            P_absorbing=P_absorbing)
    # stormpy.export_to_drn(builderS.imdp, 'out.drn')
    print(f'- Build with storm took: {(time.time() - t):.3f} sec.')

    t = time.time()
    builderS.compute_reach_avoid()
    print(f'- Verify with storm took: {(time.time() - t):.3f} sec.')

if args.checker == 'prism' or args.debug:

    t = time.time()
    builderP = BuilderPrism(states=partition.regions['idxs'],
                            goal_regions=partition.goal['idxs'],
                            critical_regions=partition.critical['idxs'],
                            actions=actions.backreach['idxs'],
                            enabled_actions=enabled_actions,
                            P_full=P_full,
                            P_absorbing=P_absorbing)
    print(f'- Build with prism took: {(time.time() - t):.3f} sec.')

    t = time.time()
    builderP.compute_reach_avoid(args.prism_dir)
    print(f'- Verify with prism took: {(time.time() - t):.3f} sec.')

if args.debug:
    assert np.all(np.abs(builderS.results - builderP.results)) < 1e-4