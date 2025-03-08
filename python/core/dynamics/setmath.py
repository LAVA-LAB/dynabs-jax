import numpy as np
import jax.numpy as jnp
import jax

@jax.jit
def box(x_min, x_max):
    # Check if x_min < x_max and if not, flip around.
    return jnp.minimum(x_min, x_max), jnp.maximum(x_min, x_max)

@jax.jit
def mult(X, Y):
    # Multiply two boxes with each other
    x_min, x_max = X
    y_min, y_max = Y

    x_min, x_max = box(x_min, x_max)
    y_min, y_max = box(y_min, y_max)

    # Multiply all combinations
    x_min_y_min = x_min * y_min
    x_max_y_min = x_max * y_min
    x_min_y_max = x_min * y_max
    x_max_y_max = x_max * y_max

    Z = jnp.vstack((x_min_y_min, x_max_y_min, x_min_y_max, x_max_y_max))
    z_min = jnp.min(Z, axis=0)
    z_max = jnp.max(Z, axis=0)

    return z_min, z_max

@jax.jit
def sin(x_min, x_max):
    x_min, x_max = box(x_min, x_max)

    # Shift such that x_min is always in [0,2*pi]
    mod = x_min % (jnp.pi * 2)
    x_min -= mod * jnp.pi
    x_max -= mod * jnp.pi

    # If 0.5*pi is in the interval, then the minimum is 1
    y_min = 1 * (x_max > 0.5*jnp.pi) + jnp.minimum(jnp.sin(x_min),jnp.sin(x_max)) * (x_max <= 0.5*jnp.pi)

    # If 1.5*pi is in the interval, then the minimum is -1
    y_max = 1 * (x_max > 1.5 * jnp.pi) + jnp.maximum(jnp.sin(x_min),jnp.sin(x_max)) * (x_max <= 1.5 * jnp.pi)

    return y_min, y_max

@jax.jit
def cos(x_min, x_max):
    x_min, x_max = box(x_min, x_max)

    # Shift such that x_min is always in [0,2*pi]
    mod = x_min % (jnp.pi*2)
    x_min -= mod * jnp.pi
    x_max -= mod * jnp.pi

    # If pi is in the interval, then the minimum is -1
    y_min = -1 * (x_max > jnp.pi) + jnp.minimum(jnp.cos(x_min), jnp.cos(x_max)) * (x_max <= jnp.pi)

    # If 2*pi is in the interval, then the maximum is 1
    y_max = 1 * (x_max > 2*jnp.pi) + jnp.maximum(jnp.cos(x_min), jnp.cos(x_max)) * (x_max <= 2*jnp.pi)

    return y_min, y_max