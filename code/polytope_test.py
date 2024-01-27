import jax.numpy as jnp
import numpy as np
import jax
import time

class polytope:

    def __init__(self, A, b):

        self.A = A
        self.b = b

@jax.jit
def points_in_polytope_jit(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (jnp.matmul(A, points.T) <= b)

    # A point is contained if every constraint is satisfied
    points_contained = jnp.all(bools, axis=0)

    return points_contained

def points_in_polytope(A, b, points):
    ''' Check if polytope defined by Ax <= b contains given list of points '''

    # Check matrix inequality
    bools = (A @ points.T <= b)

    # A point is contained if every constraint is satisfied
    points_contained = np.all(bools, axis=0)

    return points_contained

A = jnp.array([
    [1, 0],
    [0, -1],
    [1, 0],
    [0, -1]
], dtype = jnp.float32)

b = jnp.array([
    [3],
    [-(-1)],
    [2],
    [-(-2)]
], dtype=jnp.float32)

points = jnp.array([
    [0, 0],
    [4, 0]
], dtype=jnp.float32)

contained = points_in_polytope(A, b, points)

reps = 100000

t = time.time()
for i in range(reps):
    contained1 = points_in_polytope(A, b, points)
print(f'Normal took {(time.time()-t):.3f}')

contained = points_in_polytope_jit(A, b, points)

t = time.time()
for i in range(reps):
    contained2 = points_in_polytope_jit(A, b, points)
print(f'Jitted took {(time.time()-t):.3f}')

A_stacked = np.repeat(np.array([A]), 100*reps, axis=0)
b_stacked = np.repeat(np.array([b]), 100*reps, axis=0)
print('Matrices stacked..')

vfun = jax.vmap(points_in_polytope_jit, in_axes=(0, 0, None), out_axes=0)

t = time.time()
contained3 = vfun(A_stacked, b_stacked, points)
print(f'Vectorized took {(time.time()-t):.3f}')

