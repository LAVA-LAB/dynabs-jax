import datetime
import os
import time
from pathlib import Path

import jax
import numpy as np

import benchmarks
from core.actions_forward import RectangularForward
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition
from core.prob_Gauss import compute_probabilities_per_dim

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
args.model = 'Dubins_small'
args.model_version = 1

# In debug mode, configure jax to use Float64 (for more accurate computations)
if args.debug:
    from jax import config

    config.update("jax_enable_x64", True)

# Set current working directory
args.cwd = os.path.dirname(os.path.abspath(__file__))
args.root_dir = Path(args.cwd)

stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f'Run started at {stamp} using arguments:')
for key, val in vars(args).items():
    print(' - `' + str(key) + '`: ' + str(val))
print('\n==============================\n')

# Define and parse model
if args.model == 'Dubins':
    base_model = benchmarks.Dubins(args)
elif args.model == 'Dubins_small':
    base_model = benchmarks.Dubins_small(args)
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

# Create actions based on forward reachable sets
actions = RectangularForward(partition=partition, model=model)

# With forward reachability, every action is enabled in every state
enabled_actions = np.full((len(partition.regions['centers']), len(actions.idxs)), fill_value=True, dtype=np.bool)

print(f"(Number of actions in each state: {np.sum(np.any(enabled_actions, axis=0))})\n")

P_full, P_id, _, P_absorbing, _ = compute_probabilities_per_dim(args, model, partition, actions.frs, actions.max_slice)

# %% Model checking

from core.imdp import BuilderStorm

# Compute optimal policy on the iMDP abstraction
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

print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')
print(builderS.imdp)

t = time.time()
result = builderS.compute_reach_avoid()
policy, policy_inputs = builderS.get_policy(actions)
print(f'- Verify with storm took: {(time.time() - t):.3f} sec.')
print('Total sum of reach probs:', np.sum(builderS.results))
print('In state {}: {}'.format(model.x0, builderS.get_value_from_tuple(model.x0, partition)))

# %% Simulations and plot

from core.simulate import MonteCarloSim
from plotting.traces import plot_traces
from plotting.heatmap import heatmap

sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=10000)
print('Empirical satisfaction probability:', sim.results['satprob'])

plot_traces(stamp, [0, 1], partition, model, sim.results['traces'], line=True, num_traces=4)
heatmap(stamp, idx_show=[0, 1], slice_values=np.zeros(model.n), partition=partition, results=builderS.results)
