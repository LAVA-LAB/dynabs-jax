from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from core import setmath


def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins_small(object):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        self.linear = False
        self.set_model(args)
        self.set_spec()
        print('')

    def set_model(self, args):
        '''
        Set model parameters.

        :param args: Arguments object.
        '''

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        # Discretization step size
        self.tau = 1

        self.n = 3
        self.p = 2

        self.alpha_min = 0.80
        self.alpha_max = 0.90
        self.alpha = 0.85

        # self.beta_min = 0.80
        # self.beta_max = 0.90

        self.state_variables = ['x', 'y', 'angle']
        self.wrap = jnp.array([False, False, True], dtype=bool)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['cov'] = np.diag([0, 0, 0.1])
        self.noise['cov_diag'] = np.array([0, 0, 0.1])

        return

    def step(self, state, action, noise):
        '''
        Make a step under the dynamics.

        :param state: Current state.
        :param action: Control input that is executed.
        :param noise: Realization of the stochastic process noise.
        :return: Next state.
        '''

        [x, y, theta] = state
        [u1, u2] = action
        x_next = x + self.tau * u2 * np.cos(theta)
        y_next = y + self.tau * u2 * np.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])

        state_next = jnp.array([x_next, y_next, theta_next])
        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        '''
        Compute the forward reachable set for the box of states [state_min, state_max] under the control input [action_min, action_max].

        :param state_min: Lower bound state.
        :param state_max: Upper bound state.
        :param action_min: Lower bound control input.
        :param action_max: Upper bound control input.
        :return: Forward reachable set represented as a box.
        '''

        # Convert to boxes
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min] = state_min
        [x_max, y_max, theta_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = action_min
        [u1_max, u2_max] = action_max

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))

        state_next = jnp.vstack((x_next,  # jnp.clip(x_next, self.partition['boundary_jnp'][0][0] + 1e-3, self.partition['boundary_jnp'][1][0] - 1e-3),
                                 y_next,  # jnp.clip(y_next, self.partition['boundary_jnp'][0][1] + 1e-3, self.partition['boundary_jnp'][1][1] - 1e-3),
                                 theta_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([20, 20, 11])

        self.goal = np.array([
            [[5, 5, -np.pi], [10, 10, np.pi]]
        ], dtype=float)

        self.critical = np.array([
            [[-10, -10, -np.pi], [-9, -9, np.pi]],
        ], dtype=float)

        self.x0 = np.array([-5, 5, 0])

        return
