import argparse


def parse_arguments():
    """
    Function to parse arguments provided

    Returns
    -------
    :args: Object with all arguments

    """

    # Options
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, perform additional checks to debug python")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generators (Jax, Numpy)")
    parser.add_argument('--decimals', type=int, default=4,
                        help="Number of decimals to work with for storing probabilities")

    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run on GPU. Otherwise, run on CPU")

    parser.add_argument('--num_samples', type=int, default=1000,
                        help="Number of samples to compute probability intervals for")
    parser.add_argument('--confidence', type=float, default=0.01,
                        help="Confidence level on each individual transition probability")

    parser.add_argument('--model', type=str, default='Drone2D',
                        help="Benchmark model to run")
    parser.add_argument('--model_version', type=int, default=0,
                        help="Version of the model to use (optinal; 0 by default)")
    parser.add_argument('--checker', type=str, default='storm',
                        help="Model checker to use (prism or storm)")
    parser.add_argument('--prism_dir', type=str, default='~/Documents/Tools/prism/prism/bin/prism',
                        help="Directory where Prism is located")

    parser.add_argument('--mode', type=str, default='fori_loop',
                        help="Should be one of 'fori_loop', 'vmap', 'python'")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for functions vectorized with Jax")

    # Parse arguments
    args = parser.parse_args()

    return args
