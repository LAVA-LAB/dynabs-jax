#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from pathlib import Path

from plotting.utils import plot_boxes, plot_grid, set_plot_lims
from core.utils import cm2inch, remove_consecutive_duplicates
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_traces(idx_show, partition, model, traces, line=True, num_traces=10, folder=False, filename=False, add_unsafe_box=True):
    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5), dpi=300)

    font = {'family': 'normal',
            'size': 10}
    mpl.rc('font', **font)

    i1, i2 = np.array(idx_show, dtype=int)

    plt.xlabel(f'${model.state_variables[i1]}$', labelpad=0)
    plt.ylabel(f'${model.state_variables[i2]}$', labelpad=0)

    if add_unsafe_box:
        expand = 1
    else:
        expand = 0

    set_plot_lims(ax,
                  state_min=np.array(partition.boundary_lb)[[i1,i2]] - expand,
                  state_max=np.array(partition.boundary_ub)[[i1,i2]] + expand,
                  width=np.array(partition.cell_width))

    # Plot grid
    plot_grid(ax,
              state_min=np.array(partition.boundary_lb)[[i1, i2]],
              state_max=np.array(partition.boundary_ub)[[i1, i2]],
              size=[1, 1])

    # Plot goal/unsafe regions
    plot_boxes(ax, model,plot_dimensions=[i1,i2])

    # Plot boundary of unsafe regions if requested
    if add_unsafe_box:
        state_lb = np.array(partition.boundary_lb)
        state_ub = np.array(partition.boundary_ub)


        LOWS = [np.array([state_lb[i1] - expand, state_lb[i2] - expand]),
                np.array([state_lb[i1], state_lb[i2] - expand]),
                np.array([state_lb[i1], state_ub[i2]]),
                np.array([state_ub[i1], state_lb[i2] - expand])
                ]
        HIGHS = [np.array([state_lb[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1], state_lb[i2]]),
                 np.array([state_ub[i1], state_ub[i2] + expand]),
                 np.array([state_ub[i1] + expand, state_ub[i2] + expand]),
                ]

        for low, high in zip(LOWS, HIGHS):
            width, height = (high - low)

            print(low, width, height)

            ax.add_patch(Rectangle(low, width, height, facecolor='red', alpha=0.3, edgecolor='red'))

    # Add traces
    i = 0
    for trace in traces.values():
        state_trace = np.array(trace['x'])[:, [i1, i2]]

        # Only show trace if there are at least two points
        if len(state_trace) < 2:
            continue
        else:
            i += 1

        # Stop at desired number of traces
        if i > num_traces:
            break

        state_trace = remove_consecutive_duplicates(state_trace)

        # Plot precise points
        plt.plot(*state_trace.T, 'o', markersize=1, color="black");

        if line:
            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(state_trace, axis=0) ** 2,
                                                axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)

            if len(state_trace) == 2:
                kind = 'linear'
            else:
                kind = 'quadratic'

            interpolator = interp1d(distance, state_trace, kind=kind,
                                    axis=0)
            interpolated_points = interpolator(alpha)

            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);

    plt.gca().set_aspect('equal')

    # Set tight layout
    fig.tight_layout()

    # Save figure
    plt.savefig('traces.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('traces.png', format='png', bbox_inches='tight')

    plt.show()


def UAV_plot_2D(i_show, setup, args, regions, goal_regions, critical_regions,
                spec, traces, cut_idx, traces_to_plot=10, line=False):
    '''
    Create 2D trajectory plots for the 2D UAV benchmark

    '''

    from scipy.interpolate import interp1d

    i_show = np.array(i_show, dtype=int)
    is1, is2 = i_show
    i_hide = np.array([i for i in range(len(spec.partition['width']))
                       if i not in i_show], dtype=int)

    print('Show state variables', i_show, 'and hide', i_hide)

    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))

    plt.xlabel('$x$', labelpad=0)
    plt.ylabel('$y$', labelpad=0)

    width = np.array(spec.partition['width'])
    domainMax = width * np.array(spec.partition['number']) / 2

    min_xy = spec.partition['origin'] - domainMax
    max_xy = spec.partition['origin'] + domainMax

    major_ticks_x = np.arange(min_xy[is1] + 1, max_xy[is1] + 1, 4 * width[is1])
    major_ticks_y = np.arange(min_xy[is2] + 1, max_xy[is2] + 1, 4 * width[is2])

    minor_ticks_x = np.arange(min_xy[is1], max_xy[is1] + 1, width[is1])
    minor_ticks_y = np.arange(min_xy[is2], max_xy[is2] + 1, width[is2])

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)

    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)

    plt.grid(which='minor', color='#CCCCCC', linewidth=0.3)

    # Goal x-y limits
    ax.set_xlim(min_xy[is1], max_xy[is1])
    ax.set_ylim(min_xy[is2], max_xy[is2])

    ax.set_title("N = " + str(args.noise_samples), fontsize=10)

    keys = list(regions['idx'].keys())
    # Draw goal states
    for goal in goal_regions:

        goalIdx = np.array(keys[goal])

        if all(goalIdx[i_hide] == cut_idx):
            goal_lower = [regions['low'][goal][is1], regions['low'][goal][is2]]
            goalState = Rectangle(goal_lower, width=width[is1],
                                  height=width[is2], color="green",
                                  alpha=0.3, linewidth=None)
            ax.add_patch(goalState)

    keys = list(regions['idx'].keys())
    # Draw critical states
    for crit in critical_regions:

        critIdx = np.array(keys[crit])

        if all(critIdx[i_hide] == cut_idx):
            critStateLow = [regions['low'][crit][is1], regions['low'][crit][is2]]
            criticalState = Rectangle(critStateLow, width=width[is1],
                                      height=width[is2], color="red",
                                      alpha=0.3, linewidth=None)
            ax.add_patch(criticalState)

    # Add traces
    i = 0
    for trace in traces.values():

        state_traj = trace['x']

        # Only show trace if there are at least two points
        if len(state_traj) < 2:
            printWarning('Warning: trace ' + str(i) +
                         ' has length of ' + str(len(state_traj)))
            continue
        else:
            i += 1

        # Stop at desired number of traces
        if i >= traces_to_plot:
            break

        # Convert nested list to 2D array
        trace_array = np.array(state_traj)

        # Extract x,y coordinates of trace
        x = trace_array[:, is1]
        y = trace_array[:, is2]
        points = np.array([x, y]).T

        # Plot precise points
        plt.plot(*points.T, 'o', markersize=1, color="black");

        if line:

            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2,
                                                axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # Interpolation for different methods:
            alpha = np.linspace(0, 1, 75)

            if len(state_traj) == 2:
                kind = 'linear'
            else:
                kind = 'quadratic'

            interpolator = interp1d(distance, points, kind=kind,
                                    axis=0)
            interpolated_points = interpolator(alpha)

            # Plot trace
            plt.plot(*interpolated_points.T, '-', color="blue", linewidth=1);

    # Set tight layout
    fig.tight_layout()

    # Save figure
    plt.savefig('traces.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('traces.png', format='png', bbox_inches='tight')

    plt.show()