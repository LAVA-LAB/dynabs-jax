import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap(stamp, idx_show, slice_values, partition, results):
    '''
    Create heat map for the satisfaction probability from any initial state.

    Parameters
    ----------

    Returns
    -------
    None.

    '''

    i1, i2 = np.array(idx_show, dtype=int)

    lb = np.array(partition.boundary_lb)
    ub = np.array(partition.boundary_ub)

    values = np.zeros((partition.number_per_dim[i2], partition.number_per_dim[i1]))
    slice_idx = np.array(((slice_values - lb) / (ub - lb) * np.array(partition.number_per_dim)) // 1, dtype=int)

    # Fill values in matrix to plot in heatmap
    for x in range(partition.number_per_dim[i1]):
        for y in range(partition.number_per_dim[i2]):
            slice_at = slice_idx
            slice_at[i1] = x
            slice_at[i2] = y

            # Retrieve state ID
            state_idx = partition.region_idx_array[tuple(slice_at)]

            # Fill heatmap value
            values[y, x] = results[state_idx]

    X = partition.regions_per_dim['centers'][i1]
    Y = partition.regions_per_dim['centers'][i2]

    DF = pd.DataFrame(values[::-1, :], index=Y[::-1], columns=X)

    sns.heatmap(DF)

    # Save figure
    plt.savefig(f'output/heatmap_{stamp}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'output/heatmap_{stamp}.png', format='png', bbox_inches='tight')

    plt.show()
