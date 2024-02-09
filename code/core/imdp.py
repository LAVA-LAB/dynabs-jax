import logging
import pycarl
import stormpy
import numpy as np

def imdp_test():
    """
    Construct a hardcoded iMDP to test storm(py)
    """

    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                       has_custom_row_grouping=True, row_groups=0)

    builder.new_row_group(0)
    builder.add_next_value(0, 1, pycarl.Interval(0.5, 0.7))
    builder.add_next_value(0, 4, pycarl.Interval(0.35, 0.47))

    builder.new_row_group(1)
    builder.add_next_value(1, 0, pycarl.Interval(0.1, 0.4))
    builder.add_next_value(1, 2, pycarl.Interval(0.2, 0.3))
    builder.add_next_value(1, 3, pycarl.Interval(0.5, 0.7))

    builder.new_row_group(2)
    builder.add_next_value(2, 2, pycarl.Interval(1, 1))

    builder.new_row_group(3)
    builder.add_next_value(3, 2, pycarl.Interval(0.6, 0.7))
    builder.add_next_value(3, 4, pycarl.Interval(0.3, 0.5))
    builder.add_next_value(4, 0, pycarl.Interval(1, 1))

    builder.new_row_group(5)
    builder.add_next_value(5, 4, pycarl.Interval(1, 1))

    matrix = builder.build()
    logging.debug(matrix)

    # Create state labeling
    state_labeling = stormpy.storage.StateLabeling(5)

    state_labeling.add_label('init')
    state_labeling.add_label_to_state('init', 0)

    state_labeling.add_label('goal')
    state_labeling.add_label_to_state('goal', 2)

    components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
    imdp = stormpy.storage.SparseIntervalMdp(components)

    prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

    task = stormpy.CheckTask(prop.raw_formula, only_initial_states=True)
    task.set_produce_schedulers()
    task.set_robust_uncertainty(True)

    results = stormpy.check_interval_mdp(imdp, task, env)

    result_init = results.at(0)

    return matrix, imdp, result_init

def imdp_test2():
    """
    Construct a hardcoded iMDP to test storm(py)
    """

    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                       has_custom_row_grouping=True, row_groups=0)

    builder.new_row_group(0)
    builder.add_next_value(0, 1, pycarl.Interval(0.995831, 1))
    builder.add_next_value(0, 2, pycarl.Interval(0, 0.00416917))

    builder.add_next_value(1, 1, pycarl.Interval(0.995831, 1))
    builder.add_next_value(1, 2, pycarl.Interval(0, 0.00416917))

    builder.add_next_value(2, 1, pycarl.Interval(0.995831, 1))
    builder.add_next_value(2, 2, pycarl.Interval(0, 0.00416917))

    builder.add_next_value(3, 1, pycarl.Interval(0.995831, 1))
    builder.add_next_value(3, 2, pycarl.Interval(0, 0.00416917))

    builder.new_row_group(4)
    builder.add_next_value(4, 1, pycarl.Interval(1, 1))

    builder.new_row_group(5)
    builder.add_next_value(5, 2, pycarl.Interval(1, 1))

    matrix = builder.build()
    logging.debug(matrix)

    # Create state labeling
    state_labeling = stormpy.storage.StateLabeling(3)

    state_labeling.add_label('init')
    state_labeling.add_label_to_state('init', 0)

    state_labeling.add_label('goal')
    state_labeling.add_label_to_state('goal', 1)

    components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
    imdp = stormpy.storage.SparseIntervalMdp(components)

    prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

    task = stormpy.CheckTask(prop.raw_formula, only_initial_states=True)
    task.set_produce_schedulers()
    task.set_robust_uncertainty(True)

    results = stormpy.check_interval_mdp(imdp, task, env)

    result_init = results.at(0)

    return matrix, imdp, result_init

class Builder:
    """
    Construct iMDP
    """

    def __init__(self, states, goal_regions, critical_regions, actions, enabled_actions, P_full, P_absorbing):

        self.builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                           has_custom_row_grouping=True, row_groups=0)

        # Set some constants
        self.goal_state = np.max(states) + 1
        self.critical_state = np.max(states) + 2
        self.absorbing_state = np.max(states) + 3

        # Predefine the pycarl intervals
        self.intervals = {}
        # self.intervals_goal = {}
        # self.intervals_critical = {}
        self.intervals_absorbing = {}
        for a in actions:
            self.intervals[a] = {}

            # Add intervals for each successor state
            for ss in states:
                if P_full[a, ss, 0] > 0: # and ss not in critical_regions and ss not in goal_regions:
                    self.intervals[a][ss] = pycarl.Interval(P_full[a, ss, 0], P_full[a, ss, 1])

            # Add intervals for other states
            # self.intervals_goal[a] = pycarl.Interval(P_goal[a, 0], P_goal[a, 1])
            # self.intervals_critical[a] = pycarl.Interval(P_critical[a, 0], P_critical[a, 1])
            self.intervals_absorbing[a] = pycarl.Interval(P_absorbing[a, 0], P_absorbing[a, 1])

        row = 0
        states_created = 0

        # For all states
        for s in states:
            print(f'- State {s}')

            # For each state, create a new row group
            self.builder.new_row_group(row)
            states_created += 1
            enabled_in_s = np.where(enabled_actions[s])[0]

            # If no actions are enabled at all, add a deterministic transition to the absorbing state
            if len(enabled_in_s) == 0 or s in critical_regions or s in goal_regions:
                self.builder.add_next_value(row, s, pycarl.Interval(1, 1))
                row += 1

            else:

                # For every enabled action
                for a in enabled_in_s:

                    # Add transitions for current (s,a) pair to other normal states
                    for ss, intv in self.intervals[a].items():
                        self.builder.add_next_value(row, ss, intv)

                    # Add transitions to other states
                    # self.builder.add_next_value(row, self.goal_state, self.intervals_goal[a])
                    # self.builder.add_next_value(row, self.critical_state, self.intervals_critical[a])
                    self.builder.add_next_value(row, self.absorbing_state, self.intervals_absorbing[a])

                    # For each (s,a) pair, increment the row count by one
                    row += 1

        for ss in [self.goal_state, self.critical_state, self.absorbing_state]:
            self.builder.new_row_group(row)
            self.builder.add_next_value(row, ss, pycarl.Interval(1, 1))
            row += 1
            states_created += 1

        print(f'Number of rows created: {row}')
        print(f'Number of states created: {states_created}')
        self.nr_states = states_created

        matrix = self.builder.build()
        logging.debug(matrix)

        # Create state labeling
        state_labeling = stormpy.storage.StateLabeling(self.nr_states)

        # Define initial states
        state_labeling.add_label('init')
        state_labeling.add_label_to_state('init', 667)

        # Add absorbing (unsafe) states
        state_labeling.add_label('absorbing')
        state_labeling.add_label_to_state('absorbing', self.absorbing_state)

        # Add critical (unsafe) states
        state_labeling.add_label('critical')
        state_labeling.add_label_to_state('critical', self.critical_state)
        for s in critical_regions:
            state_labeling.add_label_to_state('critical', s)

        # Add goal states
        state_labeling.add_label('goal')
        state_labeling.add_label_to_state('goal', self.goal_state)
        for s in goal_regions:
            state_labeling.add_label_to_state('goal', s)

        components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
        self.imdp = stormpy.storage.SparseIntervalMdp(components)

    def compute_reach_avoid(self, maximizing=True):

        prop = stormpy.parse_properties('P{}=? [F "goal"]'.format('max' if maximizing else 'min'))[0]
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

        # Compute reach-avoid probability
        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=True)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)
        self.result_robust = stormpy.check_interval_mdp(self.imdp, task, env)

        # initial_state = self.imdp.initial_states[0]

        # self.prob_robust = result_robust.at(initial_state)
        # self.scheduler_robust = result_robust.scheduler