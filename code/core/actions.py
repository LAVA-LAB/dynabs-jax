import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
import cdd

@jax.jit
def backward_reach(target, A_inv, B, Q_flat, input_vertices):

    inner = target - jnp.matmul(B, input_vertices.T).T - Q_flat
    vertices = jnp.matmul(A_inv, inner.T).T

    return vertices

# Vectorized function over different sets of points
vmap_backward_reach = jax.vmap(backward_reach, in_axes=(0, None, None, None, None), out_axes=0)

def compute_polytope_halfsaces(vertices):
    '''Compute the halfspace representation (H-rep) of a polytope.'''
    t = np.ones((vertices.shape[0], 1))  # first column is 1 for vertices
    tV = np.hstack([t, vertices])
    mat = cdd.Matrix(tV, number_type="float")
    mat.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(mat)
    bA = np.array(P.get_inequalities())

    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])

    return A,b

class RectangularTarget(object):

    def __init__(self, target_points, model):
        print('Define target points and backward reachable sets...')

        t = time.time()
        self.target_points = target_points
        vertices = vmap_backward_reach(target_points, model.A_inv, model.B, model.Q_flat, model.uVertices)
        print(f'- Backward reachable sets computed (took {(time.time()-t):.3f} sec.)')

        t = time.time()
        A = [[]] * len(vertices)
        b = [[]] * len(vertices)
        for i,vertices in enumerate(vertices):
            A[i], b[i] = compute_polytope_halfsaces(vertices)
        print(f'- Halfspace representations computed (took {(time.time() - t):.3f} sec.)')

        self.backreach = {
            'idxs': np.arange(len(vertices)),
            'vertices': vertices,
            'A': np.array(A),
            'b': np.array(b)
        }

        print('')
        return

# Vectorized function over different polytopes
from .polytope import all_points_in_polytope
vmap_all_points_in_polytope = jax.jit(jax.vmap(all_points_in_polytope, in_axes=(0, 0, None), out_axes=0))

def compute_enabled_actions(As, bs, region_vertices, mode = 'fori_loop'):
    print('Compute subset of enabled actions in each partition element...')

    @jax.jit
    def loop_body(i, val):
        As, bs, vertices, bools = val
        bool = vmap_all_points_in_polytope(As, bs, vertices[i])
        bools = bools.at[i].set(bool)
        return (As, bs, vertices, bools)

    t = time.time()
    if mode == 'fori_loop':

        enabled_actions = jnp.full((len(region_vertices), len(As)), fill_value=False)
        val = (As, bs, region_vertices, enabled_actions)
        val = jax.lax.fori_loop(0, len(region_vertices), loop_body, val)
        (_, _, _, enabled_actions) = val

    elif mode == 'vmap':

        vmap_compute_actions_enabled_in_region = jax.jit(
            jax.vmap(vmap_all_points_in_polytope, in_axes=(None, None, 0), out_axes=0))

        enabled_actions = vmap_compute_actions_enabled_in_region(As, bs, region_vertices)

    elif mode == 'pmap':

        pmap_compute_actions_enabled_in_region = jax.jit(
            jax.pmap(vmap_all_points_in_polytope, in_axes=(None, None, 0), out_axes=0, devices=jax.devices('cpu')))

        enabled_actions = pmap_compute_actions_enabled_in_region(As, bs, region_vertices)

    else:

        enabled_actions = np.array([vmap_all_points_in_polytope(As,bs,vertices) for vertices in region_vertices])

    print(f'- Enabled actions computed (took {(time.time() - t):.3f} sec.)')

    print('')
    return np.array(enabled_actions)