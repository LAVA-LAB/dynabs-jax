import numpy as np
import scipy
import jax.numpy as jnp
from core.dynamics import setmath
import jax
from functools import partial

class Dubins(object):

    def __init__(self):
        '''
        Defines the 2D drone benchmark, with a 4D LTI system
        '''

        self.linear = False

        self.set_model()
        self.set_spec()

    def set_model(self):
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        # Discretization step size
        self.tau = 1.0

        self.n = 4
        self.p = 2

        self.alpha_min = 0.8
        self.alpha_max = 1.0

        self.beta_min = 0.8
        self.beta_max = 0.9

        self.state_variables = ['x', 'y', 'angle', 'velocity']
        self.wrap = jnp.array([False, False, True, False], dtype=bool)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['cov'] = np.diag([0, 0, 0.1, 0])

        return

    @jax.jit
    def step_no_noise(self, state, action):

        [x, y, theta, V] = state
        [u1, u2] = action
        x_next = x + self.tau * V * self.cosFunc(theta)
        y_next = y + self.tau * V * self.sinFunc(theta)
        theta_next = theta + self.tau * self.alpha * u1
        V_next = self.beta * V + self.tau * u2

        state_next = jnp.array([x_next, y_next, theta_next, V_next])
        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):

        # Convert to boxes
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min, V_min] = state_min
        [x_max, y_max, theta_max, V_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = action_min
        [u1_max, u2_max] = action_max

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))
        V_next = jnp.concat(setmath.mult([self.beta_min, self.beta_max], [V_min, V_max])) + jnp.array([u2_min, u2_max])

        state_next = jnp.vstack((x_next, y_next, theta_next, V_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

    def set_spec(self):
        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-2, -2,]
        self.uMax = [2, 2]
        self.num_actions = [5,5]

        self.partition['boundary'] = np.array([[-10, -10, -np.pi, -1], [10, 10, np.pi, 1]])
        self.partition['number_per_dim'] = np.array([20, 20, 10, 10])

        self.goal = np.array([
            [[-2, 2, -np.pi, -0.1], [0.1, 0.1, np.pi, 0.1]]
        ], dtype=float)

        self.critical = np.array([
            [[-10, -10, -np.pi, -1], [-9, -9, np.pi, 1]],
        ], dtype=float)

        return
