import numpy as np
import jax.numpy as jnp
import jax
import os
import time
from pathlib import Path
from core.options import parse_arguments
from core.model import parse_model
from core.partition import RectangularPartition
from core.actions import RectangularTarget, compute_enabled_actions
from core.probabilities import sample_noise, count_samples_per_region, compute_scenario_interval_table, \
    samples_to_intervals
from core.imdp import BuilderStorm, BuilderPrism

import benchmarks

args = parse_arguments()
np.random.seed(args.seed)
args.jax_key = jax.random.PRNGKey(args.seed)

# In debug mode, configure jax to use Float64 (for more accurate computations)
if args.debug:
    from jax import config

    config.update("jax_enable_x64", True)

# Set current working directory
args.cwd = os.path.dirname(os.path.abspath(__file__))
args.root_dir = Path(args.cwd)

print('Run using arguments:')
for key, val in vars(args).items():
    print(' - `' + str(key) + '`: ' + str(val))
print('\n==============================\n')

# Define and parse model
if args.model == 'Drone2D':
    base_model = benchmarks.Drone2D()
elif args.model == 'Drone3D':
    base_model = benchmarks.Drone3D()
elif args.model == 'Spacecraft':
    base_model = benchmarks.Spacecraft()
else:
    assert False, f"The passed model '{args.model}' could not be found"

# Parse given model
model = parse_model(base_model)

# Create partition of the continuous state space into convex polytope
partition = RectangularPartition(model=model)
print(f"(Number of states: {len(partition.regions['idxs'])})\n")

# Create actions (compute backward reachable set for all target points)
actions = RectangularTarget(target_points=partition.regions['centers'], model=model)

# If debug is enabled, test the correctness of one (arbitrary) backward reachable set
if args.debug:
    actions.test_backwardset(idx=10, model=model)

print(f"(Number of actions: {len(actions.target_points)})\n")

# Compute the enabled actions in each state
# TODO: Investigate using a dense matrix here (generally, it will be very sparse)
enabled_actions = compute_enabled_actions(As=jnp.array(actions.backreach['A']),
                                          bs=jnp.array(actions.backreach['b']),
                                          region_vertices=np.array(partition.regions['all_vertices']),
                                          mode='vmap',
                                          batch_size=args.batch_size)

print(f"(Number of enabled actions: {np.sum(np.any(enabled_actions, axis=0))})\n")

# Compute noise samples and count the number of samples in every partition element
samples = sample_noise(model, args.jax_key, args.num_samples)
# TODO: Investigate using a dense matrix here (generally, it will be very sparse)
num_samples_per_state = count_samples_per_region(args, model, partition, actions.backreach['target_points'],
                                                 samples, mode='vmap', batch_size=args.batch_size)

# Load scenario approach table with probability intervals for the given number of samples and confidence level
table_filename = f'intervals_N={args.num_samples}_beta={args.confidence}.csv'
interval_table = compute_scenario_interval_table(Path(str(args.root_dir), 'interval_tables', table_filename),
                                                 args.num_samples, args.confidence)

# Compute probability intervals
P_full, P_absorbing = samples_to_intervals(args.num_samples,
                                           num_samples_per_state,
                                           interval_table,
                                           round_probabilities=False)

# Compute optimal policy on the iMDP abstraction
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

    print(builderS.imdp)

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

# If debugging is enabled, compare if the results from Storm and Prism match
if args.debug:
    assert np.all(np.abs(builderS.results - builderP.results)) < 1e-4
