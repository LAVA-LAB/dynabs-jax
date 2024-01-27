import jax
import jax.numpy as jnp


def batched_forward_pass(apply_fn, batch_size, ):



    if len(samples) <= batch_size:
        # If the number of samples is below the maximum batch size, then just do one pass
        return jit(apply_fn)(jax.lax.stop_gradient(params), jax.lax.stop_gradient(samples))

    else:
        # Otherwise, split into batches
        output = np.zeros((len(samples), out_dim))
        num_batches = np.ceil(len(samples) / batch_size).astype(int)
        starts = np.arange(num_batches) * batch_size
        ends = np.minimum(starts + batch_size, len(samples))

        for (i, j) in zip(starts, ends):
            output[i:j] = jit(apply_fn)(jax.lax.stop_gradient(params), jax.lax.stop_gradient(samples[i:j]))

        return output