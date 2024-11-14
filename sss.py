import Funcs as Fs
import numpy as np

# Parameters
rate_LH = 0.25
rate_HL = 0.0025
total_cells = 10 ** 6
mutation_rate_L = 10 ** -8
mutation_rate_H = 10 ** -4
initial_cells = 1
num_colonies = 10 ** 3
generations = 20

# Calculate initial probabilities
prob_LH = Fs.fl_hat(rate_LH, rate_HL)
prob_HL = Fs.fh_hat(rate_LH, rate_HL)

# Initialize arrays to store mutation counts and colony states
mutation_counts = np.zeros(num_colonies)
colony_states = np.zeros((num_colonies, 2))

# Initialize colonies
for i in range(num_colonies):
    colony_states[i] = Fs.init_col(prob_LH, initial_cells)

# Simulate over generations
for t in range(generations):
    for i in range(num_colonies):
        mutation_counts[i] += Fs.m_tot(total_cells, colony_states[i, 0], colony_states[i, 1], mutation_rate_H, mutation_rate_L)
        colony_states[i] = Fs.next_state(colony_states[i, 0], colony_states[i, 1], rate_HL, rate_LH)

# Calculate and print expected value and variance of mutations
expected_mutations = np.mean(mutation_counts)
variance_mutations = np.var(mutation_counts)
print(f'E(M) = {expected_mutations}, V(M) = {variance_mutations}')
print(f'V/E = {variance_mutations / expected_mutations}')