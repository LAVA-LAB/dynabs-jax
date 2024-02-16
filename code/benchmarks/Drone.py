import numpy as np
import scipy

class Drone2D(object):

    def __init__(self):
        '''
        Defines the 2D drone benchmark, with a 4D LTI system
        '''

        self.set_model()
        self.set_spec()

    def set_model(self):

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                           [0, 1]])

        # Input matrix
        Bblock = np.array([[self.tau ** 2 / 2],
                           [self.tau]])

        self.A = scipy.linalg.block_diag(Ablock, Ablock)
        self.B = scipy.linalg.block_diag(Bblock, Bblock)

        # Disturbance matrix
        self.Q = np.array([[0], [0], [0], [0]])

        self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel']

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A, 1)) * 0.15

        return

    def set_spec(self):

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-4, -4]
        self.uMax = [4, 4]

        self.partition['boundary'] = np.array([[-7, -3, -7, -3], [7, 3, 7, 3]]) * 2
        self.partition['number_per_dim'] = np.array([7, 4, 7, 4]) * 2

        self.goal = np.array([
            [[5, -3, 5, -3], [7, 3, 7, 3]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, -3, 1, -3], [-1, 3, 3, 3]],
            [[3, -3, -7, -3], [7, 3, -3, 3]],
        ], dtype=float)

        return



class Drone3D(object):

    def __init__(self):
        '''
        Defines the 2D drone benchmark, with a 4D LTI system
        '''

        self.set_model()
        self.set_spec()

    def set_model(self):

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 2

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                           [0, 1]])

        # Input matrix
        Bblock = np.array([[self.tau ** 2 / 2],
                           [self.tau]])

        self.A = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
        self.B = scipy.linalg.block_diag(Bblock, Bblock, Bblock)

        # Disturbance matrix
        self.Q = np.array([[0], [0], [0], [0], [0], [0]])

        self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel', 'z_pos', 'z_vel']

        # Covariance of the process noise
        self.noise = dict()
        self.noise['w_cov'] = np.eye(np.size(self.A, 1)) * 0.15

        return

    def set_spec(self):

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-4, -4, -4]
        self.uMax = [4, 4, 4]

        self.partition['boundary'] = np.array([[-15, -2.25, -9, -2.25, -7, -2.25],
                                               [15, 2.25, 9, 2.25, 7, 2.25]])
        self.partition['number_per_dim'] = np.array([15, 3, 9, 3, 7, 3])

        self.goal = np.array([
            [[11, -3, 1, -3, -7, -3], [15, 3, 1, 3, -7, 3]]
        ], dtype=float)

        l = -3
        u = 3

        self.critical = np.array([
            # Hole 1
            [[-11, l, -1, l, -7, l], [-5, u, 9, u, -5, u]],
            [[-11, l, 5, l, -5, l], [-5, u, 9, u, 5, u]],
            [[-11, l, -1, l, -5, l], [-5, u, 3, u, 3, u]],
            # Hole 2
            [[-1, l, 1, l, -7, l], [3, u, 9, u, -1, u]],
            [[-1, l, 1, l, 3, l], [3, u, 9, u, 5, u]],
            [[-1, l, 1, l, -1, l], [3, u, 3, u, 3, u]],
            [[-1, l, 7, l, -1, l], [3, u, 9, u, 3, u]],
            # Tower
            [[-1, l, -3, l, -7, l], [3, u, 1, u, 7, u]],
            # Wall between routes
            [[3, l, -3, l, -7, l], [9, u, 1, u, -1, u]],
            # Long route obstacles
            [[-11, l, -5, l, -7, l], [-7, u, -1, u, 1, u]],
            [[-1, l, -9, l, -7, l], [3, u, -3, u, -5, u]],
            # Overhanging
            [[-1, l, -9, l, 3, l], [3, u, -3, u, 7, u]],
            # Small last obstacle
            [[11, l, -9, l, -7, l], [15, u, -5, u, -5, u]],
            # Obstacle next to goal
            [[9, l, 5, l, -7, l], [15, u, 9, u, 1, u]],
        ], dtype=float)

        return