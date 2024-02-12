import stormpy
import pydot
import pycarl

print('\nThis version does not work:')
builder = stormpy.IntervalSparseMatrixBuilder(rows=4, columns=3, entries=0, force_dimensions=True,
                                                   has_custom_row_grouping=True, row_groups=0)

state_labeling = stormpy.storage.StateLabeling(3)
state_labeling.add_label('init')
state_labeling.add_label('goal')

builder.new_row_group(0)
builder.add_next_value(0, 0, pycarl.Interval(1, 1))
state_labeling.add_label_to_state('goal', 0)

builder.new_row_group(1)
builder.add_next_value(1, 1, pycarl.Interval(1, 1))

builder.new_row_group(2)
builder.add_next_value(2, 0, pycarl.Interval(0.3, 0.3))
builder.add_next_value(2, 1, pycarl.Interval(0.7, 0.7))
builder.add_next_value(3, 0, pycarl.Interval(0.3, 0.3))
builder.add_next_value(3, 1, pycarl.Interval(0.7, 0.7))
state_labeling.add_label_to_state('init', 2)

matrix = builder.build()
print(matrix)

components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
imdp = stormpy.storage.SparseIntervalMdp(components)

prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
env = stormpy.Environment()
env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
task.set_robust_uncertainty(True)

results = stormpy.check_interval_mdp(imdp, task, env)

for s in range(3):
    print(f'Result at {s} = {results.at(s)}')

stormpy.export_to_drn(imdp, 'out1.drn')

dot = imdp.to_dot()
graph = pydot.graph_from_dot_data(dot)
graph[0].write_png('graph1.png')

# %%

print('\nThis version does work:')
builder = stormpy.IntervalSparseMatrixBuilder(rows=4, columns=4, entries=0, force_dimensions=True,
                                                   has_custom_row_grouping=True, row_groups=0)

state_labeling = stormpy.storage.StateLabeling(4)
state_labeling.add_label('init')
state_labeling.add_label('goal')

builder.new_row_group(0)
builder.add_next_value(0, 2, pycarl.Interval(0.3, 0.4))
builder.add_next_value(0, 3, pycarl.Interval(0.4, 0.9))
state_labeling.add_label_to_state('init', 0)

builder.new_row_group(1)
builder.add_next_value(1, 0, pycarl.Interval(0.3, 0.4))
builder.add_next_value(1, 3, pycarl.Interval(0.4, 0.9))
state_labeling.add_label_to_state('init', 1)

builder.new_row_group(2)
builder.add_next_value(2, 2, pycarl.Interval(1, 1))
state_labeling.add_label_to_state('goal', 2)

builder.new_row_group(3)
builder.add_next_value(3, 3, pycarl.Interval(1, 1))

matrix = builder.build()
print(matrix)

components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
imdp = stormpy.storage.SparseIntervalMdp(components)

prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
env = stormpy.Environment()
env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
task.set_robust_uncertainty(True)

results = stormpy.check_interval_mdp(imdp, task, env)

for s in range(4):
    print(f'Result at {s} = {results.at(s)}')

stormpy.export_to_drn(imdp, 'out2.drn')

dot = imdp.to_dot()
graph = pydot.graph_from_dot_data(dot)
graph[0].write_png('graph3.png')

# %%

print('\nThis version does not work:')
builder = stormpy.IntervalSparseMatrixBuilder(rows=4, columns=4, entries=0, force_dimensions=True,
                                                   has_custom_row_grouping=True, row_groups=0)

state_labeling = stormpy.storage.StateLabeling(4)
state_labeling.add_label('init')
state_labeling.add_label('goal')

builder.new_row_group(0)
builder.add_next_value(0, 1, pycarl.Interval(0.3, 0.4))
builder.add_next_value(0, 3, pycarl.Interval(0.4, 0.9))
state_labeling.add_label_to_state('init', 0)

builder.new_row_group(1)
builder.add_next_value(1, 1, pycarl.Interval(1, 1))
state_labeling.add_label_to_state('goal', 1)

builder.new_row_group(2)
builder.add_next_value(2, 0, pycarl.Interval(0.3, 0.4))
builder.add_next_value(2, 3, pycarl.Interval(0.4, 0.9))
state_labeling.add_label_to_state('init', 2)

builder.new_row_group(3)
builder.add_next_value(3, 3, pycarl.Interval(1, 1))

matrix = builder.build()
print(matrix)

components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
imdp = stormpy.storage.SparseIntervalMdp(components)

prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
env = stormpy.Environment()
env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
task.set_robust_uncertainty(True)

results = stormpy.check_interval_mdp(imdp, task, env)

for s in range(4):
    print(f'Result at {s} = {results.at(s)}')

stormpy.export_to_drn(imdp, 'out2.drn')

dot = imdp.to_dot()
graph = pydot.graph_from_dot_data(dot)
graph[0].write_png('graph4.png')