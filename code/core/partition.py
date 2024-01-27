import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
from tqdm import tqdm

EPS = 1e-3

@jax.jit
def meshgrid_jax(points, size):
    '''
        Set rectangular grid

        :param low: ndarray
        :param high: ndarray
        :param size: List of ints (entries per dimension)
        '''

    meshgrid = jnp.asarray(jnp.meshgrid(*points))
    grid = jnp.reshape(meshgrid, (len(size), -1)).T

    return grid

def define_grid_jax(low, high, size):
    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid

@jax.jit
def center2halfspace(center, cell_width):
    ''' From given centers and cell widths, compute the halfspace inequalities Ax <= b. '''

    A1 = jnp.identity(len(center))
    A2 = -jnp.identity(len(center))

    b1 = center + cell_width / 2
    b2 = -(center - cell_width / 2)

    A = jnp.concatenate((A1, A2))
    b = jnp.concatenate((b1, b2))

    return A,b



# Vectorized function over different polytopes
from .polytope import points_in_polytope
vmap_points_in_polytope = jax.jit(jax.vmap(points_in_polytope, in_axes=(0, 0, None), out_axes=0))

@jax.jit
def check_if_region_in_goal(goals_A, goals_b, points):

    # Vectorized over all goal regions
    points_contained = vmap_points_in_polytope(goals_A, goals_b, points)

    # Check for every goal region if all points are contained in the polytope
    all_points_contained = jnp.all(points_contained, axis=1)

    # If any goal region is contained in the polytope, then set current polytope as goal
    return jnp.any(all_points_contained)

# Vectorized function over different sets of points
vmap_check_if_region_in_goal = jax.vmap(check_if_region_in_goal, in_axes=(None, None, 0), out_axes=0)

@jax.jit
def get_vertices_from_bounds(lb, ub):
    # Stack lower and upper bounds in one array
    stacked = jnp.vstack((lb, ub))

    # Get all vertices (by taking combinations of lower and upper bounds)
    vertices = meshgrid_jax(stacked.T, lb)

    return vertices

class RectangularPartition(object):

    def __init__(self, number_per_dim, partition_boundary, goal_regions, critical_regions, mode = 'fori_loop'):

        print('Define rectangular partition...')

        t = time.time()
        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5
        ub_center = partition_boundary[1] - self.cell_width * 0.5

        # Define the grid centers
        centers = define_grid_jax(lb_center, ub_center, number_per_dim)
        region_idxs = np.arange(len(centers))
        lower_bounds = centers - self.cell_width / 2
        upper_bounds = centers + self.cell_width / 2

        # Determine the vertices of all partition elements
        vmap_get_vertices_from_bounds = jax.vmap(get_vertices_from_bounds, in_axes=(0, 0), out_axes=0)
        all_vertices = vmap_get_vertices_from_bounds(lower_bounds, upper_bounds)
        print(f'- Grid points defined (took {(time.time()-t):.3f} sec.)')
        
        t = time.time()
        # Determine halfspace (Ax <= b) inequalities
        vmap_center2halfspace = jax.vmap(center2halfspace, in_axes=(0, None), out_axes=(0, 0))
        all_A, all_b = vmap_center2halfspace(centers, self.cell_width)
        print(f'- Halfspace inequalities (Ax <= b) defined (took {(time.time()-t):.3f} sec.)')

        self.regions = {
            'centers': centers,
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'all_vertices': all_vertices,
            'A': all_A,
            'b': all_b
        }

        t = time.time()
        # Compute halfspace representation of the goal regions
        goal_centers = np.zeros((len(goal_regions), len(number_per_dim)))
        goal_widths  = np.zeros((len(goal_regions), len(number_per_dim)))
        for i,goal in enumerate(goal_regions):
            goal_centers[i] = (goal[1] + goal[0]) / 2
            goal_widths[i] = (goal[1] - goal[0]) + EPS

        vmap_center2halfspace = jax.vmap(center2halfspace, in_axes=(0, 0), out_axes=(0, 0))
        goals_A, goals_b = vmap_center2halfspace(goal_centers, goal_widths)

        # Determine goal regions
        goal_regions_bools = vmap_check_if_region_in_goal(goals_A, goals_b, all_vertices)
        goal_regions_idxs = region_idxs[goal_regions_bools]
        goal_regions_centers = centers[goal_regions_bools]
        print(f'- Goal regions defined (took {(time.time()-t):.3f} sec.)')

        self.goal = {
            'bools': goal_regions_bools,
            'idxs': goal_regions_idxs,
            'centers': goal_regions_centers
        }

        t = time.time()
        # Determine critical regions by sampling within critical regions and check if any points are contained
        # TODO: Improve by removing the sampling step
        samples = [[]]*len(critical_regions)
        for i,critical in enumerate(critical_regions):
            # Size must be chosen small enough to ensure correct computation
            size = np.array((critical[1] - critical[0]) / (0.5 * self.cell_width) + 1, dtype=int)
            samples[i] = define_grid_jax(critical[0] + EPS, critical[1] - EPS, size)
        critical_samples = jnp.concatenate(samples)
        print(f'- Samples for computing critical regions defined (took {(time.time()-t):.3f} sec.)')

        self.critical_samples = critical_samples

        @jax.jit
        def loop_body(i, val):
            As, bs, critical_samples, bools = val
            bool = any_points_in_polytope(As[i], bs[i], critical_samples)
            bools = bools.at[i].set(bool)
            return (As, bs, critical_samples, bools)

        # Check for each element of the partition if any critical sample is contained (if so, it's a critical region)
        t = time.time()

        if mode == 'fori_loop':

            critical_regions_bools = jnp.full(len(self.regions['A']), fill_value=True)
            val = (self.regions['A'], self.regions['b'], critical_samples, critical_regions_bools)
            val = jax.lax.fori_loop(0, len(self.regions['A']), loop_body, val)
            (_, _, _, critical_regions_bools) = val

        elif mode == 'vmap':

            from .polytope import any_points_in_polytope
            vmap_any_points_in_polytope = jax.jit(jax.vmap(any_points_in_polytope, in_axes=(0, 0, None), out_axes=0))

            critical_regions_bools = vmap_any_points_in_polytope(self.regions['A'], self.regions['b'],
                                                                 critical_samples)

        elif mode == 'pmap':

            from .polytope import any_points_in_polytope
            pmap_any_points_in_polytope = jax.jit(jax.pmap(any_points_in_polytope, in_axes=(0, 0, None), out_axes=0), devices=jax.devices('cpu'))

            critical_regions_bools = pmap_any_points_in_polytope(self.regions['A'], self.regions['b'],
                                                                 critical_samples)

        else:

            critical_regions_bools = np.array([any_points_in_polytope(A, b, critical_samples)
                                               for A,b in zip(self.regions['A'], self.regions['b'])])

        critical_regions_idxs = region_idxs[critical_regions_bools]
        critical_regions_centers = centers[critical_regions_bools]
        print(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

        self.critical = {
            'bools': critical_regions_bools,
            'idxs': critical_regions_idxs,
            'centers': critical_regions_centers
        }

        print('')
        return