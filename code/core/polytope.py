import jax
import jax.numpy as jnp
from functools import partial

def points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.all(bools, axis=1)

    return points_contained

def any_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.min(bools, axis=1) #jnp.all(bools, axis=1)

    return jnp.max(points_contained) #jnp.any(points_contained)

def all_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T <= b)

    # A point is contained if every constraint is satisfied
    # points_contained = jnp.all(bools, axis=1)
    # return jnp.all(points_contained)

    return jnp.min(bools)

def num_points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T).T < b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.min(bools, axis=1) #jnp.all(bools, axis=1)

    return jnp.sum(points_contained)