from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from pathlib import Path


def heatmap(idx_show, slice_values, partition, model, results, title='auto'):
    '''
    Create heat map for the reachability probability from any initial state.

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
            values[y,x] = results[state_idx]

    X = partition.regions_per_dim['centers'][i1]
    Y = partition.regions_per_dim['centers'][i2]

    DF = pd.DataFrame(values[::-1, :], index=Y[::-1], columns=X)

    sns.heatmap(DF)

    # Save figure
    plt.savefig('heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('heatmap.png', format='png', bbox_inches='tight')

    plt.show()