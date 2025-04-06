import numpy as np
import scipy
import jax.numpy as jnp
from core.dynamics import setmath
import jax
from functools import partial

def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

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
        self.tau = 0.5

        self.n = 4
        self.p = 2

        mode = 2

        if mode == 0:
            # No parameter uncertainty
            self.alpha_min = 0.85
            self.alpha_max = 0.85
            self.alpha = 0.85

            self.beta_min = 0.85
            self.beta_max = 0.85
            self.beta = 0.85
        elif mode == 1:
            # Low parameter uncertainty
            self.alpha_min = 0.85
            self.alpha_max = 0.90
            self.alpha = 0.88

            self.beta_min = 0.85
            self.beta_max = 0.90
            self.beta = 0.88
        elif mode == 2:
            # High parameter uncertainty
            self.alpha_min = 0.80
            self.alpha_max = 0.90
            self.alpha = 0.81

            self.beta_min = 0.80
            self.beta_max = 0.90
            self.beta = 0.89
        elif mode == 3:
            # High parameter uncertainty
            self.alpha_min = 0.75
            self.alpha_max = 0.95
            self.alpha = 0.85

            self.beta_min = 0.75
            self.beta_max = 0.95
            self.beta = 0.85
        else:
            # High parameter uncertainty
            self.alpha_min = 0.7
            self.alpha_max = 1.0
            self.alpha = 0.85

            self.beta_min = 0.7
            self.beta_max = 1.0
            self.beta = 0.85

        self.state_variables = ['x', 'y', 'angle', 'velocity']
        self.wrap = jnp.array([False, False, True, False], dtype=bool)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['cov'] = np.diag([0, 0, 0.1, 0])
        self.noise['cov_diag'] = np.array([0, 0, 0.1, 0])

        return

    # @jax.jit
    # def step_no_noise(self, state, action):
    #
    #     [x, y, theta, V] = state
    #     [u1, u2] = action
    #     x_next = x + self.tau * V * self.cosFunc(theta)
    #     y_next = y + self.tau * V * self.sinFunc(theta)
    #     theta_next = theta + self.tau * self.alpha * u1
    #     V_next = self.beta * V + self.tau * u2
    #
    #     state_next = jnp.array([x_next, y_next, theta_next, V_next])
    #     return state_next

    def step(self, state, action, noise):

        [x, y, theta, V] = state
        [u1, u2] = action
        x_next = x + self.tau * V * np.cos(theta)
        y_next = y + self.tau * V * np.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])
        V_next = self.beta * V + self.tau * u2

        state_next = jnp.array([x_next,
                                y_next,
                                theta_next,
                                np.clip(V_next, self.partition['boundary_jnp'][0][3] + 1e-3, self.partition['boundary_jnp'][1][3] - 1e-3)])
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

        Vmean = (V_max + V_min) / 2
        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))
        V_next = jnp.concat(setmath.mult([self.beta_min, self.beta_max], [V_min, V_max])) + self.tau * jnp.array([u2_min, u2_max])

        state_next = jnp.vstack((x_next, #jnp.clip(x_next, self.partition['boundary_jnp'][0][0] + 1e-3, self.partition['boundary_jnp'][1][0] - 1e-3),
                                 y_next, #jnp.clip(y_next, self.partition['boundary_jnp'][0][1] + 1e-3, self.partition['boundary_jnp'][1][1] - 1e-3),
                                 theta_next,
                                 jnp.clip(V_next, self.partition['boundary_jnp'][0][3] + jnp.array([1e-3, 2e-3]), self.partition['boundary_jnp'][1][3] - jnp.array([2e-3, 1e-3]))))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

    def set_spec(self):
        self.partition = {}
        self.targets = {}

        layout = 3

        if layout == 1:

            # Authority limit for the control u, both positive and negative
            self.uMin = [-0.5*np.pi, -5]
            self.uMax = [0.5*np.pi, 5]
            self.num_actions = [6, 6]

            self.partition['boundary'] = np.array([[0, 0, -np.pi, -5], [10, 10, np.pi, 5]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([20, 20, 20, 20])

            self.goal = np.array([
                [[6, 6, -2*np.pi, -10], [9, 9, 2*np.pi, 10]]
            ], dtype=float)

            self.critical = np.array([
                [[-5, 0, -2 * np.pi, -10], [-4, 5, 2 * np.pi, 10]],
                [[4, 5, -2 * np.pi, -10], [5, 10, 2 * np.pi, 10]],
                # [[-1, 5, -np.pi, -5], [1, 10, np.pi, 5]],
            ], dtype=float)

            self.x0 = np.array([2, 8, 0, 0])

        elif layout == 2:

            # Authority limit for the control u, both positive and negative
            self.uMin = [-0.5 * np.pi, -5]
            self.uMax = [0.5 * np.pi, 5]
            self.num_actions = [7, 7]

            self.partition['boundary'] = np.array([[0, 0, -np.pi, -3], [10, 10, np.pi, 3]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([20, 20, 20, 20])

            self.goal = np.array([
                [[5, 5, -np.pi, -3], [10, 10, np.pi, 3]]
            ], dtype=float)

            self.critical = np.array([
                [[-10, -10, -np.pi, -3], [-9, -9, np.pi, 3]],
            ], dtype=float)

            self.x0 = np.array([2, 8, 0, 0])

        else:

            # Authority limit for the control u, both positive and negative
            self.uMin = [-0.5 * np.pi, -5]
            self.uMax = [0.5 * np.pi, 5]
            self.num_actions = [7, 7]

            self.partition['boundary'] = np.array([[-10, 0, -np.pi, -3], [10, 10, np.pi, 3]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([40, 20, 20, 20])

            self.goal = np.array([
                [[6, 6, -np.pi, -3], [9, 9, np.pi, 3]]
            ], dtype=float)

            # self.critical = np.array([
            #     # [[-10, -10, -np.pi, -3], [-9, -9, np.pi, 3]],
            #     [[4, 5, -2 * np.pi, -3], [5, 10, 2 * np.pi, 3]],
            #     # [[4, 4, -2 * np.pi, -3], [6, 5, 2 * np.pi, 3]],
            #     [[-1, 0, -2 * np.pi, -3], [0, 5, 2 * np.pi, 3]],
            #     [[-6, 4, -2 * np.pi, -3], [-1, 5, 2 * np.pi, 3]],
            #     [[-10, 8, -2 * np.pi, -3], [-8, 10, 2 * np.pi, 3]],
            #     [[-3, 5, -2 * np.pi, -3], [-2, 6, 2 * np.pi, 3]],
            # ], dtype=float)

            self.critical = np.array([
                # [[-10, -10, -np.pi, -3], [-9, -9, np.pi, 3]],
                [[4, 5, -2 * np.pi, -3], [5, 10, 2 * np.pi, 3]],
                # [[4, 4, -2 * np.pi, -3], [6, 5, 2 * np.pi, 3]],
                [[-1, 0, -2 * np.pi, -3], [0, 5, 2 * np.pi, 3]],
                [[-5, 4, -2 * np.pi, -3], [-1, 5, 2 * np.pi, 3]],
                # [[-10, 8, -2 * np.pi, -3], [-8, 10, 2 * np.pi, 3]],
                # [[-3, 5, -2 * np.pi, -3], [-2, 7, 2 * np.pi, 3]],
            ], dtype=float)

            self.x0 = np.array([-2.5, 2.5, 0, 0])

        return
