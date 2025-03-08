import numpy as np
import scipy
import jax.numpy as jnp

class Spacecraft(object):

    def __init__(self):
        '''
        Defines the 2D drone benchmark, with a 4D LTI system
        '''

        self.linear = True

        self.set_model()
        self.set_spec()

    def set_model(self):
        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2

        # Discretization step size
        self.tau = 1.0
        T = self.tau

        mu = 3.986e14 / 1e9
        r0 = 42164
        n = np.sqrt(mu / r0 ** 3)

        self.A = np.array([
            [4 - 3 * np.cos(n * T), 0, 0, 1 / n * np.sin(n * T), 2 / n * (1 - np.cos(n * T)), 0],
            [6 * (np.sin(n * T) - n * T), 1, 0, -2 / n * (1 - np.cos(n * T)), 1 / n * (4 * np.sin(n * T) - 3 * n * T),
             0],
            [0, 0, np.cos(n * T), 0, 0, 1 / n * np.sin(n * T)],
            [3 * n * np.sin(n * T), 0, 0, np.cos(n * T), 2 * np.sin(n * T), 0],
            [-6 * n * (1 - np.cos(n * T)), 0, 0, -2 * np.sin(n * T), 4 * np.cos(n * T) - 3, 0],
            [0, 0, -n * np.sin(n * T), 0, 0, np.cos(n * T)]
        ])

        self.B = np.array([
            [1 / n * np.sin(n * T), 2 / n * (1 - np.cos(n * T)), 0],
            [-2 / n * (1 - np.cos(n * T)), 1 / n * (4 * np.sin(n * T) - 3 * n * T), 0],
            [0, 0, 1 / n * np.sin(n * T)],
            [np.cos(n * T), 2 * np.sin(n * T), 0],
            [-2 * np.sin(n * T), 4 * np.cos(n * T) - 3, 0],
            [0, 0, np.cos(n * T)]
        ])

        self.n, self.p = self.B.shape

        # Disturbance matrix
        self.q = np.zeros(6)

        self.state_variables = ['x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel']
        self.wrap = jnp.array([False, False, False, False, False, False], dtype=bool)

        # Covariance of the process noise
        self.noise = dict()
        self.noise['cov'] = np.diag([.1, .1, .01, .01, .01, .01])

        return

    def set_spec(self):
        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-2, -2, -2]
        self.uMax = [2, 2, 2]

        self.partition['boundary'] = np.array([[-3.4, -2, -2, -4, -4, -2],
                                               [1, 16.4, 2, 4, 4, 2]])
        self.partition['number_per_dim'] = np.array([11, 23, 5, 5, 5, 5])

        self.goal = np.array([
            [[-0.2, -0.4, -0.4, -4, -4, -2], [0.2, 0.4, 0.4, 4, 4, 2]]
        ], dtype=float)

        self.critical = np.array([
            [[-1, 12.4 - 8 * 0.8, -2, -4, -4, -2], [1, 12.4, 2, 4, 4, 2]],
        ], dtype=float)

        return
