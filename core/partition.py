import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
from tqdm import tqdm
from .polytope import hyperrectangles_isdisjoint_multi

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

    return A, b


# Vectorized function over different polytopes
from .polytope import points_in_polytope

vmap_points_in_polytope = jax.jit(jax.vmap(points_in_polytope, in_axes=(0, 0, None), out_axes=0))

from .polytope import any_points_in_polytope

vmap_any_points_in_polytope = jax.jit(jax.vmap(any_points_in_polytope, in_axes=(0, 0, None), out_axes=0))


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

    def __init__(self, model):
        print('Define rectangular partition...')
        t_total = time.time()

        # Retrieve necessary data from the model object
        self.number_per_dim = model.partition['number_per_dim']
        partition_boundary = model.partition['boundary']
        self.boundary_lb = partition_boundary[0]
        self.boundary_ub = partition_boundary[1]
        goal_regions = model.goal
        critical_regions = model.critical

        # Set partition as being (hyper)rectangula
        self.rectangular = True

        t = time.time()
        # From the partition boundary, determine where the first grid centers are placed
        self.cell_width = (partition_boundary[1] - partition_boundary[0]) / self.number_per_dim
        lb_center = partition_boundary[0] + self.cell_width * 0.5
        ub_center = partition_boundary[1] - self.cell_width * 0.5

        # First define a grid where each region is a unit cube
        lb_unit = jnp.zeros(len(lb_center), dtype=int)
        ub_unit = jnp.array(self.number_per_dim - 1, dtype=int)
        centers_unit = define_grid_jax(lb_unit, ub_unit, self.number_per_dim)

        # TODO: Remove this
        from .utils import lexsort4d
        centers_unit = lexsort4d(centers_unit)

        # TODO: Check how to avoid this step
        # Define n-dimensional array (n = dimension of state space) to index elements of the partition
        centers = jnp.array(centers_unit, dtype=int)
        self.region_idx_array = np.zeros(self.number_per_dim, dtype=int)
        self.region_idx_array[tuple(centers.T)] = np.arange(len(centers))
        self.region_idx_array = jnp.array(self.region_idx_array)
        # Define list with each element containing its index elements
        self.region_idx_inv = centers

        # Now scale the unit-cube partition appropriately
        centers = centers_unit * self.cell_width + lb_center

        region_idxs = jnp.arange(len(centers))
        lower_bounds = centers - self.cell_width / 2
        upper_bounds = centers + self.cell_width / 2

        # Determine the vertices of all partition elements
        vmap_get_vertices_from_bounds = jax.vmap(get_vertices_from_bounds, in_axes=(0, 0), out_axes=0)
        all_vertices = vmap_get_vertices_from_bounds(lower_bounds, upper_bounds)
        print(f'- Grid points defined (took {(time.time() - t):.3f} sec.)')

        t = time.time()
        # Determine halfspace (Ax <= b) inequalities
        vmap_center2halfspace = jax.vmap(center2halfspace, in_axes=(0, None), out_axes=(0, 0))
        all_A, all_b = vmap_center2halfspace(centers, self.cell_width)
        print(f'- Halfspace inequalities (Ax <= b) defined (took {(time.time() - t):.3f} sec.)')

        self.regions = {
            'centers': jnp.array(centers, dtype=float),
            'idxs': region_idxs,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'all_vertices': all_vertices,
            'A': all_A,
            'b': all_b
        }
        self.size = len(centers)

        # Also store the partition bounds per dimension
        elems_per_dim = [jnp.arange(num) for num in self.number_per_dim]
        centers_per_dim = [elems_per_dim[i] * self.cell_width[i] + lb_center[i] for i in range(model.n)]
        lower_bounds_per_dim = [jnp.array(centers_per_dim[i] - self.cell_width[i] / 2) for i in range(model.n)]
        upper_bounds_per_dim = [jnp.array(centers_per_dim[i] + self.cell_width[i] / 2) for i in range(model.n)]

        self.regions_per_dim = {
            'centers': centers_per_dim,
            'idxs': elems_per_dim,
            'lower_bounds': lower_bounds_per_dim,
            'upper_bounds': upper_bounds_per_dim,
        }

        t = time.time()
        if len(critical_regions) > 0:
            # Compute halfspace representation of the goal regions
            goal_centers = np.zeros((len(goal_regions), len(self.number_per_dim)))
            goal_widths = np.zeros((len(goal_regions), len(self.number_per_dim)))
            for i, goal in enumerate(goal_regions):
                goal_centers[i] = (goal[1] + goal[0]) / 2
                goal_widths[i] = (goal[1] - goal[0]) + EPS

            goal_centers = jnp.array(goal_centers, dtype=float)
            goal_widths = jnp.array(goal_widths, dtype=float)

            vmap_center2halfspace = jax.vmap(center2halfspace, in_axes=(0, 0), out_axes=(0, 0))
            goals_A, goals_b = vmap_center2halfspace(goal_centers, goal_widths)

            # Determine goal regions
            goal_regions_bools = vmap_check_if_region_in_goal(goals_A, goals_b, all_vertices)
            goal_regions_idxs = region_idxs[goal_regions_bools]
            goal_regions_centers = centers[goal_regions_bools]
        else:
            goal_regions_bools = jnp.full(self.size, False, dtype=bool)
            goal_regions_idxs = jnp.array([], dtype=int)
            goal_regions_centers = jnp.array([], dtype=float)
        print(f'- Goal regions defined (took {(time.time() - t):.3f} sec.)')

        self.goal = {
            'bools': goal_regions_bools,
            'idxs': goal_regions_idxs.tolist(),
            'centers': goal_regions_centers
        }
        print(f"-- Number of goal regions: {len(self.goal['idxs'])}")

        t = time.time()
        if len(critical_regions) > 0:
            # Check which regions (hyperrectangles) are *not* disjoint from the critical regions (also hyperrectangles)
            critical_lbs = critical_regions[:, 0, :]
            critical_ubs = critical_regions[:, 1, :]

            vfun = jax.jit(jax.vmap(hyperrectangles_isdisjoint_multi, in_axes=(0, 0, None, None), out_axes=0))
            critical_regions_bools = ~vfun(self.regions['lower_bounds'], self.regions['upper_bounds'],
                                           critical_lbs + EPS, critical_ubs - EPS)
            critical_regions_idxs = region_idxs[critical_regions_bools]
            critical_regions_centers = centers[critical_regions_bools]
        else:
            critical_regions_bools = jnp.full(self.size, False, dtype=bool)
            critical_regions_idxs = jnp.array([], dtype=int)
            critical_regions_centers = jnp.array([], dtype=float)
        print(f'- Critical regions defined (took {(time.time() - t):.3f} sec.)')

        self.critical = {
            'bools': critical_regions_bools,
            'idxs': critical_regions_idxs.tolist(),
            'centers': critical_regions_centers
        }
        print(f"-- Number of critical regions: {len(self.critical['idxs'])}")

        print(f'Partitioning took {(time.time() - t_total):.3f} sec.')
        print('')
        return

    def x2state(self, x):
        # Discard samples outside of partition
        in_partition = np.all((x >= self.boundary_lb) * (x <= self.boundary_ub))

        if in_partition:
            # Normalize samples
            x_norm = np.array(((x - self.boundary_lb) / (self.boundary_ub - self.boundary_lb) * self.number_per_dim) // 1, dtype=int)
            state = int(self.region_idx_array[tuple(x_norm)])
            return state, True

        else:
            return self.size, False

