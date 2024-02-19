import numpy as np
import jax.numpy as jnp
import jax
import os
import time
from pathlib import Path
from benchmarks.Drone import Drone2D, Drone3D
from core.options import parse_arguments
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions
from core.probabilities import sample_noise, compute_samples_per_state, compute_scenario_interval_table, \
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
base_model = Drone3D()
model = parse_model(base_model)
# args.debug= True

partition = RectangularPartition(number_per_dim=model.partition['number_per_dim'],
                                 partition_boundary=model.partition['boundary'],
                                 goal_regions=model.goal,
                                 critical_regions=model.critical,
                                 mode = 'fori_loop')
print(f"(Number of states: {len(partition.regions['idxs'])})\n")

actions = RectangularTarget(target_points=partition.regions['centers'],
                            model=model)
actions.test_backwardset(idx=10, model=model)
print(f"(Number of actions: {len(actions.target_points)})\n")

enabled_actions = compute_enabled_actions(jnp.array(actions.backreach['A']),
                                          jnp.array(actions.backreach['b']),
                                          np.array(partition.regions['all_vertices']),
                                          mode = 'python',
                                          batch_size = 1000)

samples = sample_noise(model, args.jax_key, args.num_samples)

num_samples_per_state = compute_samples_per_state(args,
                                                  model,
                                                  partition,
                                                  actions.backreach['target_points'],
                                                  samples,
                                                  mode = 'python',
                                                  batch_size=100)

table_filename = f'intervals_N={args.num_samples}_beta={args.confidence}.csv'
interval_table = compute_scenario_interval_table(Path(str(args.root_dir), 'interval_tables', table_filename),
                                                 args.num_samples, args.confidence)

# Compute probability intervals
P_full, P_absorbing = samples_to_intervals(args.num_samples,
                                           num_samples_per_state,
                                           interval_table,
                                           partition.goal['bools'],
                                           partition.critical['bools'])

# %%

if args.checker == 'storm' or args.debug:
    print('Create iMDP using storm...')

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
    print('Create iMDP using prism...')

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

# %%