import numpy as np
import jax.numpy as jnp
import jax
import itertools
import time
import cdd
from tqdm import tqdm

from .utils import create_batches

from jax import config
config.update("jax_enable_x64", True)

@jax.jit
def backward_reach(target, A_inv, B, Q_flat, input_vertices):

    inner = target - jnp.matmul(B, input_vertices.T).T - Q_flat
    vertices = jnp.matmul(A_inv, inner.T).T

    return vertices

def compute_polytope_halfspaces(vertices):
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
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_backward_reach = jax.vmap(backward_reach, in_axes=(0, None, None, None, None), out_axes=0)

        t = time.time()
        self.target_points = target_points
        vertices = vmap_backward_reach(target_points, model.A_inv, model.B, model.Q_flat, model.uVertices)
        vertices = np.round(vertices, 2)
        print(vertices.shape)
        print(f'- Backward reachable sets computed (took {(time.time()-t):.3f} sec.)')

        t = time.time()
        A = [[]] * len(vertices)
        b = [[]] * len(vertices)
        for i,verts in tqdm(enumerate(vertices)):
            A[i], b[i] = compute_polytope_halfspaces(verts)
        print(f'- Halfspace representations computed (took {(time.time() - t):.3f} sec.)')

        self.backreach = {
            'idxs': np.arange(len(vertices)),
            'vertices': vertices,
            'target_points': target_points,
            'A': np.array(A),
            'b': np.array(b)
        }

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return

    def test_backwardset(self, idx, model):
        '''
        Validate the correctness of the backward reachable set for the given index.
        :param idx:
        :param model:
        :return:
        '''

        for i,(x,u) in enumerate(zip(self.backreach['vertices'][idx], model.uVertices)):
            point = model.A @ x + model.B @ u + model.Q_flat

            assert np.all(np.isclose(point, self.target_points[idx])), \
                f"""Test for backward reachable set {idx} failed for vertex {i}:
                - Target point: {self.target_points[idx]}
                - Forward dynamics from vertex results in: {point}"""

        return

# Vectorized function over different polytopes
from .polytope import all_points_in_polytope
vmap_all_points_in_polytope = jax.jit(jax.vmap(all_points_in_polytope, in_axes=(0, 0, None), out_axes=0))
vmap_compute_actions_enabled_in_region = jax.jit(jax.vmap(vmap_all_points_in_polytope, in_axes=(None, None, 0), out_axes=0))

vmap_all_points_in_polytope2 = jax.jit(jax.vmap(all_points_in_polytope, in_axes=(None, None, 0), out_axes=0))
vmap_compute_actions_enabled_in_region2 = jax.jit(jax.vmap(vmap_all_points_in_polytope2, in_axes=(0, 0, None), out_axes=0))

def compute_enabled_actions(As, bs, region_vertices, mode = 'fori_loop', batch_size=1000):
    print('Compute subset of enabled actions in each partition element...')
    t_total = time.time()

    @jax.jit
    def loop_body(i, val):
        As, bs, vertices, bools = val
        bool = vmap_all_points_in_polytope(As, bs, vertices[i])
        bools = bools.at[i].set(bool)
        return (As, bs, vertices, bools)

    t = time.time()
    if mode == 'fori_loop':
        # Use jax fori_loop

        enabled_actions = jnp.full((len(region_vertices), len(As)), fill_value=False)
        val = (As, bs, region_vertices, enabled_actions)
        val = jax.lax.fori_loop(0, len(region_vertices), loop_body, val)
        (_, _, _, enabled_actions) = val

    else:
        # Use either vmap (if batch size > 1) or plain Python for loop
        if batch_size == 1:
            enabled_actions = np.full((len(region_vertices), len(As)), fill_value=False)
            for (i, vertices) in tqdm(enumerate(region_vertices)):
                enabled_actions[i] = vmap_all_points_in_polytope(As, bs, vertices)

        else:
            starts, ends = create_batches(len(region_vertices), batch_size)
            enabled_actions = np.full((len(region_vertices), len(As)), fill_value=False)

            for (i, j) in tqdm(zip(starts, ends), total=len(starts)):
                enabled_actions[i:j] = vmap_compute_actions_enabled_in_region(As, bs, region_vertices[i:j])

    print(f'- Enabled actions computed (took {(time.time() - t):.3f} sec.)')

    print(f'Total number of enabled actions: {np.sum(np.any(enabled_actions, axis=0))}')
    print(f'Computing enabled actions took {(time.time() - t_total):.3f} sec.')
    print('')
    return np.array(enabled_actions)