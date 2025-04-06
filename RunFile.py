import numpy as np
import jax.numpy as jnp
import jax
import os
import time
from tqdm import tqdm
from pathlib import Path
from core.options import parse_arguments
from core.model import parse_linear_model, parse_nonlinear_model
from core.partition import RectangularPartition
from core.actions_backward import RectangularBackward, compute_enabled_actions
from core.actions_forward import RectangularForward
from core.prob_sample import sample_noise, count_samples_per_region, count_rectangular_single_state, compute_scenario_interval_table, \
    samples_to_intervals, samples_to_intervals_box

import benchmarks

jax.config.update("jax_default_matmul_precision", "high")

args = parse_arguments()
if args.gpu:
    jax.config.update('jax_platform_name', 'gpu')
else:
    jax.config.update('jax_platform_name', 'cpu')

print('=== JAX STATUS ===')
print(f'Devices available: {jax.devices()}')
from jax.lib import xla_bridge
print(f'Jax runs on: {xla_bridge.get_backend().platform}')
print('==================\n')

np.random.seed(args.seed)
args.jax_key = jax.random.PRNGKey(args.seed)

# args.debug = True
args.model = 'Dubins'
# args.debug = True
# args.batch_size = 1
args.decimals = 4

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
elif args.model == 'Dubins':
    base_model = benchmarks.Dubins()
elif args.model == 'Dubins_small':
    base_model = benchmarks.Dubins_small()
else:
    assert False, f"The passed model '{args.model}' could not be found"

t = time.time()

# Parse given model
if base_model.linear:
    model = parse_linear_model(base_model)
else:
    model = parse_nonlinear_model(base_model)

# Create partition of the continuous state space into convex polytope
partition = RectangularPartition(model=model)
print(f"(Number of states: {len(partition.regions['idxs'])})\n")

if base_model.linear:
    # Create actions based on backward reachable sets
    actions = RectangularBackward(target_points=partition.regions['centers'], model=model)

    # If debug is enabled, test the correctness of one (arbitrary) backward reachable set
    if args.debug:
        actions.test_backwardset(idx=10, model=model)

    print(f"(Number of actions: {len(actions.target_points)})\n")

    # Compute the enabled actions in each state
    # TODO: Investigate using a dense matrix here (generally, it will be very sparse)
    enabled_actions = compute_enabled_actions(As=jnp.array(actions.A),
                                              bs=jnp.array(actions.b),
                                              region_vertices=np.array(partition.regions['all_vertices']),
                                              mode='vmap',
                                              batch_size=args.batch_size)

else:
    # Create actions based on forward reachable sets
    actions = RectangularForward(partition=partition, model=model)

    # With forward reachability, every action is enabled in every state
    enabled_actions = np.full((len(partition.regions['centers']), len(actions.idxs)), fill_value=True, dtype=np.bool)

print(f"(Number of enabled actions: {np.sum(np.any(enabled_actions, axis=0))})\n")

# Compute noise samples and count the number of samples in every partition element
samples = sample_noise(model, args.jax_key, args.num_samples)

# %%

from core.prob_Gauss import compute_probabilities_per_dim

if base_model.linear:
    # Load scenario approach table with probability intervals for the given number of samples and confidence level
    table_filename = f'intervals_N={args.num_samples}_beta={args.confidence}.csv'
    interval_table = compute_scenario_interval_table(Path(str(args.root_dir), 'interval_tables', table_filename),
                                                     args.num_samples, args.confidence)

    # TODO: Investigate using a dense matrix here (generally, it will be very sparse)
    num_samples_per_state = count_samples_per_region(args, model, partition, actions.target_points,
                                                     samples, mode='vmap', batch_size=args.batch_size)

    # Compute probability intervals
    P_full, P_absorbing = samples_to_intervals(args.num_samples,
                                               num_samples_per_state,
                                               interval_table,
                                               round_probabilities=True)

else:

    P_full, P_id, P_nonzero, P_absorbing, keep = compute_probabilities_per_dim(args, model, partition, actions.frs, actions.max_slice)

# %%

from core.imdp import BuilderStorm, BuilderPrism

# Compute optimal policy on the iMDP abstraction
if args.checker == 'storm' or args.debug:
    print('\nCreate iMDP using storm...')

    # Build interval MDP via StormPy
    builderS = BuilderStorm(args=args,
                            partition=partition,
                            actions=actions,
                            states=np.array(partition.regions['idxs']),
                            x0=model.x0,
                            goal_regions=np.array(partition.goal['idxs']),
                            critical_regions=np.array(partition.critical['idxs']),
                            P_full=P_full,
                            P_id=P_id,
                            P_absorbing=P_absorbing)
    # stormpy.export_to_drn(builderS.imdpfrom core.imdp , 'out.drn')
    print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')

    print(builderS.imdp)

    # del partition
    # del actions
    # del P_full
    # del P_id
    # del P_nonzero
    # del P_absorbing

    t = time.time()
    result = builderS.compute_reach_avoid()
    policy, policy_inputs = builderS.get_policy(actions)
    print(f'- Verify with storm took: {(time.time() - t):.3f} sec.')

    # builderS.print_transitions(3456, 0, actions, partition)
    # builderS.print_transitions(5555, 0, actions, partition)
    # builderS.print_transitions(4020, 0, actions, partition)

    print('Total sum of reach probs:', np.sum(builderS.results))
    print('In state {}: {}'.format(model.x0, builderS.get_value_from_tuple(model.x0, partition)))

if args.checker == 'prism' or args.debug:
    print('\nCreate iMDP using prism...')

    # TODO: Incorporate state IDs in Prism builder
    t = time.time()
    builderP = BuilderPrism(state_dependent=not model.linear,
                            states=np.array(partition.regions['idxs']),
                            goal_regions=np.array(partition.goal['idxs']),
                            critical_regions=np.array(partition.critical['idxs']),
                            actions=np.array(actions.idxs, dtype=int),
                            enabled_actions=np.array(enabled_actions, dtype=bool),
                            P_full=P_full,
                            P_absorbing=P_absorbing)
    print(f'- Build with prism took: {(time.time() - t):.3f} sec.')

    t = time.time()
    builderP.compute_reach_avoid(args.prism_dir)
    print(f'- Verify with prism took: {(time.time() - t):.3f} sec.')

# If debugging is enabled, compare if the results from Storm and Prism match
if args.debug:
    assert np.all(np.abs(builderS.results - builderP.results) < 1e-4)

# %%

from core.simulate import MonteCarloSim

sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=10000)
print('Empirical satisfaction probability:', sim.results['satprob'])

# %%

from plotting.traces import plot_traces
plot_traces([0,1], partition,model, sim.results['traces'], line=True, num_traces=4)

from plotting.heatmap import heatmap
heatmap(idx_show=[0,1], slice_values=[0,0,0,0], partition=partition, model=model, results=builderS.results)

