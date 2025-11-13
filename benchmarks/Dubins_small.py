from functools import partial

from benchmarks.models import DubinsSmallDynamics
import jax
import jax.numpy as jnp
import numpy as np

from core import setmath


def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins_small(DubinsSmallDynamics):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsSmallDynamics.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

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

        # Input L_p spaces range
        # SIM_ID = 00
        # self.epsilons = np.array([0.0,0.0])
        # # SIM_ID = 01
        # self.epsilons = np.array([0.01,0.01])
        # SIM_ID = 02
        # self.epsilons = np.array([0.02,0.02])
        # # SIM_ID = 03
        # self.epsilons = np.array([0.04,0.08])
        # # SIM_ID = 04
        # self.epsilons = np.array([0.1,0.2])
        # SIM_ID = 05
        # self.epsilons = np.array([0.15,0.3])
        # # SIM_ID = 06
        # self.epsilons = np.array([0.06,0.12])
        # SIM_ID = 07
        # self.epsilons = np.array([0.09,0.18])
        # SIM_ID = 08
        self.epsilons = np.array([0.075,0.15])

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
