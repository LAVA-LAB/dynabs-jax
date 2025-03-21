import logging
import numpy as np
import pandas as pd
from .utils import writeFile
from tqdm import tqdm
import time


class BuilderStorm:
    """
    Construct iMDP
    """

    def __init__(self, args, partition, states, x0, goal_regions, critical_regions, P_full, P_id, P_absorbing):

        import pycarl
        import stormpy

        self.builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                           has_custom_row_grouping=True, row_groups=0)

        # Set some constants
        self.absorbing_state = np.max(states) + 1

        # Predefine the pycarl intervals
        self.intervals_raw = {}
        self.intervals_raw[(1, 1)] = pycarl.Interval(1, 1)

        # Reshape all probability intervals
        print('- Generate pycarl intervals...')
        P_full_flat = np.concatenate([ np.concatenate(Ps) for Ps in P_full if len(Ps) > 0])
        P_absorbing_flat = np.concatenate(list(P_absorbing.values()))

        print('-- Probability intervals reshaped')

        P_stacked = np.vstack((P_full_flat, P_absorbing_flat))

        # t = time.time()
        # P_unique = np.round(np.unique(P_stacked, axis=0), 4)
        # print(f'- Numpy took {(time.time() - t):.3f} sec.')
        # t = time.time()
        P_unique = np.round(np.sort(pd.DataFrame(P_stacked).drop_duplicates(), axis=0), args.decimals)
        # print(f'- Pandas took {(time.time() - t):.3f} sec.')
        # assert np.all(P_unique2 == P_unique)

        print('-- Unique probability intervals determined')

        # Enumerate only over unique probability intervals
        for P in tqdm(P_unique):
            self.intervals_raw[tuple(P)] = pycarl.Interval(P[0], P[1])

        self.intervals_state = {}
        self.intervals_absorbing = {}

        print('-- Pycarl intervals created')

        print('\n- Store intervals for individual transitions...')
        for s in tqdm(states):
            self.intervals_state[s] = {}
            self.intervals_absorbing[s] = {}
            for i,a in enumerate(P_id[s].keys()):
                self.intervals_state[s][a] = {}

                # Add intervals for each successor state
                for s_next,prob in zip(P_id[s][a], P_full[s][i]):
                    self.intervals_state[s][a][s_next] = self.intervals_raw[tuple(prob)]

                # Add intervals for other states
                if P_absorbing[s][a][1] > 0:
                    self.intervals_absorbing[s][a] = self.intervals_raw[tuple(P_absorbing[s][a])]

        row = 0
        states_created = 0

        # For all states
        print('\n- Build iMDP...')
        for s in tqdm(states):

            # For each state, create a new row group
            self.builder.new_row_group(row)
            states_created += 1
            enabled_in_s = P_id[s].keys()

            # If no actions are enabled at all, add a deterministic transition to the absorbing state
            if len(enabled_in_s) == 0 or s in critical_regions:
                self.builder.add_next_value(row, self.absorbing_state, self.intervals_raw[(1, 1)])
                row += 1

            elif s in goal_regions:
                self.builder.add_next_value(row, s, self.intervals_raw[(1, 1)])
                row += 1

            else:

                # For every enabled action
                for a in enabled_in_s:
                    for s_next, intv in self.intervals_state[s][a].items():
                        self.builder.add_next_value(row, s_next, intv)

                    # Add transitions to absorbing state
                    if a in self.intervals_absorbing[s]:
                        self.builder.add_next_value(row, self.absorbing_state, self.intervals_absorbing[s][a])

                    # for ss in self.successor_states[a]:
                    #     self.builder.add_next_value(row, ss, self.intervals[tuple(P_full[a, ss])])

                    # # Add transitions to absorbing state
                    # self.builder.add_next_value(row, self.absorbing_state, self.intervals[tuple(P_absorbing[a])])

                    # # Add transitions for current (s,a) pair to other normal states
                    # for ss, intv in self.intervals[a].items():
                    #     self.builder.add_next_value(row, ss, intv)
                    #
                    # # Add transitions to other states
                    # self.builder.add_next_value(row, self.absorbing_state, self.intervals_absorbing[a])

                    # For each (s,a) pair, increment the row count by one
                    row += 1

        for s in [self.absorbing_state]:
            self.builder.new_row_group(row)
            self.builder.add_next_value(row, s, self.intervals_raw[(1, 1)])
            row += 1
            states_created += 1

        self.nr_states = states_created

        matrix = self.builder.build()
        logging.debug(matrix)

        # Create state labeling
        state_labeling = stormpy.storage.StateLabeling(self.nr_states)

        # Define initial states
        state_labeling.add_label('init')
        s_init, _ = partition.x2state(x0)
        state_labeling.add_label_to_state('init', s_init)

        # Add absorbing (unsafe) states
        state_labeling.add_label('absorbing')
        state_labeling.add_label_to_state('absorbing', self.absorbing_state)

        # Add critical (unsafe) states
        state_labeling.add_label('critical')
        for s in critical_regions:
            state_labeling.add_label_to_state('critical', s)

        # Add goal states
        state_labeling.add_label('goal')
        for s in goal_regions:
            state_labeling.add_label_to_state('goal', s)

        components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
        self.imdp = stormpy.storage.SparseIntervalMdp(components)

    def compute_reach_avoid(self, maximizing=True):

        import stormpy

        prop = stormpy.parse_properties('P{}=? [F "goal"]'.format('max' if maximizing else 'min'))[0]
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

        # Compute reach-avoid probability
        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)
        result_generator = stormpy.check_interval_mdp(self.imdp, task, env)

        self.results = np.array(result_generator.get_values())[:self.nr_states]

    def get_label(self, s):

        label = ''
        if 'goal' in self.imdp.states[s].labels:
            label += 'goal,'
        elif 'absorbing' in self.imdp.states[s].labels:
            label += 'absorbing,'
        elif 'critical' in self.imdp.states[s].labels:
            label += 'critical,'

        return label

    def print_transitions(self, state, action, actions, partition):

        if type(state) in [list,tuple]:
            state = int(partition.region_idx_array[tuple(state)])

        print('\n----------')

        print('From state {} at position {} (label: {}), with action {} with inputs {}:'.format(state, partition.region_idx_inv[state], self.get_label(state), action, actions.inputs[action]))
        print(' - Optimal value: {}'.format(self.results[state]))
        print(' ---')
        print(' - State lower bound: {}'.format(partition.regions['lower_bounds'][state]))
        print(' - State upper bound: {}'.format(partition.regions['upper_bounds'][state]))
        print(' ---')
        print(' - Action FRS lower bound: {}'.format(actions.frs[state]['lb'][action]))
        print(' - Action FRS upper bound: {}'.format(actions.frs[state]['ub'][action]))
        print(' ---')

        try:
            SA = self.imdp.states[state].actions[action]

            for transition in SA.transitions:
                s_prime = transition.column
                if s_prime < len(partition.region_idx_inv):
                    idx = partition.region_idx_inv[s_prime]
                else:
                    idx = '<out-partition>'
                print(" --- With probability {}, go to state {} at position {} (label: {})".format(transition.value(), s_prime, idx, self.get_label(s_prime)))

        except:
            print(' - Error: This action does not exist in this state')

        print('----------')

    def get_value_from_tuple(self, x, region_idx_inv):
        s = region_idx_inv[tuple(x)]
        return self.results[s]

class BuilderPrism:
    """
    Construct iMDP
    """

    def __init__(self, state_dependent, states, goal_regions, critical_regions, actions, enabled_actions, P_full, P_absorbing):

        self.out_dir = 'output/'
        self.export_name = 'model'
        self.out_path = str(self.out_dir + self.export_name)
        self.PRISM_allfile = str(self.out_path + '.all')

        # Set some constants
        self.absorbing_state = np.max(states) + 1

        print('- Writing PRISM states file')

        ### Write states file
        PRISM_statefile = str(self.out_path + ".sta")

        # Define tuple of state variables (for header in PRISM state file)
        state_var_string = ['(s)']
        state_file_content = [f'{str(i)}:({str(i)})' for i in states] + \
                             [f'{str(self.absorbing_state)}:({str(self.absorbing_state)})']
        state_file_string = '\n'.join(state_var_string + state_file_content)

        # Write content to file
        writeFile(PRISM_statefile, 'w', state_file_string)

        print('- Writing PRISM label file')

        ### Write label file
        PRISM_labelfile = str(self.out_path + ".lab")

        label_head = ['0="init" 1="deadlock" 2="reached" 3="critical"']
        label_body = ['' for i in states]
        for i in states:
            substring = str(i) + ': 0'

            # Check if region is a deadlock state
            if len(enabled_actions[i]) == 0:
                substring += ' 1'

            # Check if region is in goal set
            if i in goal_regions:
                substring += ' 2'
            elif i in critical_regions:
                substring += ' 3'

            label_body[i] = substring

        label_body += [f'{str(self.absorbing_state)}: 1 3']
        label_full = '\n'.join(label_head) + '\n' + '\n'.join(label_body)

        # Write content to file
        writeFile(PRISM_labelfile, 'w', label_full)

        print('- Writing PRISM transition file')

        ### Write transition file
        PRISM_transitionfile = str(self.out_path + ".tra")

        transition_file_list = ['' for i in states]
        states_created = 0
        nr_choices_absolute = 0
        nr_transitions_absolute = 0

        # For every state
        for s in tqdm(states):
            states_created += 1
            choice = 0
            enabled_in_s = np.where(enabled_actions[s])[0]

            if state_dependent:
                Pf = P_full[s]
                Pa = P_absorbing[s]
            else:
                Pf = P_full
                Pa = P_absorbing

            # If no actions are enabled at all, add a deterministic transition to the absorbing state
            if len(enabled_in_s) == 0 or s in critical_regions or s in goal_regions:

                selfloop_prob = '[1.0,1.0]'
                substring = [f'{s} {choice} {self.absorbing_state} {selfloop_prob}']

                nr_choices_absolute += 1
                nr_transitions_absolute += 1

            else:

                substring = ['' for i in range(len(enabled_in_s))]

                # For every enabled action
                for a_idx, a in enumerate(enabled_in_s):
                    # Define name of action
                    actionLabel = "a_" + str(a)

                    # Absorbing state transition
                    str_main = [f'{s} {choice} {ss} [{Pf[a, ss, 0]},{Pf[a, ss, 1]}] {actionLabel}'
                                for ss in states if Pf[a, ss, 1] > 0]

                    if Pa[a, 1] > 0:
                        str_abs = [
                            f'{s} {choice} {self.absorbing_state} [{Pa[a, 0]},{Pa[a, 1]}] {actionLabel}']
                    else:
                        str_abs = []

                    # Increase choice counter
                    choice += 1
                    nr_choices_absolute += 1
                    nr_transitions_absolute += len(str_main) + len(str_abs)

                    # Join strings
                    substring[a_idx] = '\n'.join(str_main + str_abs)

            transition_file_list[s] = substring

        # Add one choice and transition in the absorbing state
        transition_file_list += [[f'{self.absorbing_state} 0 {self.absorbing_state} [1,1] loop']]
        states_created += 1
        nr_choices_absolute += 1
        nr_transitions_absolute += 1

        flatten = lambda t: [item for sublist in t
                             for item in sublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))

        # Header contains nr of states, choices, and transitions
        header = str(states_created) + ' ' + str(nr_choices_absolute) + ' ' + str(nr_transitions_absolute) + '\n'

        print('iMDP statistics:')
        print('- # states:', states_created)
        print('- # choices:', nr_choices_absolute)
        print('- # transitions:', nr_transitions_absolute)

        # Write content to file
        writeFile(PRISM_transitionfile, 'w', header + transition_file_list)

        ### Write specification file
        self.specification = 'Pmaxmin=? [F "reached" ]'
        specfile = str(self.out_path + ".pctl")

        # Write specification file
        writeFile(specfile, 'w', self.specification)

        self.nr_states = states_created

    def compute_reach_avoid(self, prism_folder, maximizing=True):

        import subprocess  # Import to call prism via terminal command

        policy_file = str(self.out_dir + 'policy.txt')
        vector_file = str(self.out_dir + 'vector.csv')

        options = ' -exportstrat "' + policy_file + '"' + \
                  ' -exportvector "' + vector_file + '"'

        command = f"{prism_folder} -javamaxmem 2g -importmodel '{self.PRISM_allfile}' -pf '{self.specification}' {options}"
        subprocess.Popen(command, shell=True).wait()

        values = pd.read_csv(vector_file, header=None).iloc[:self.nr_states].to_numpy()
        self.results = values.flatten()
