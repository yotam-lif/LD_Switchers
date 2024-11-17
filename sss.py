import Funcs as Fs
import numpy as np

# Parameters
rate_LH = 0.025
rate_HL = 0.25
L = 10 ** 6
mu_L = 10 ** -10
mu_H = 10 ** -6
N_0 = 1
num_colonies = 10 ** 4
generations = 20

# Calculate initial probabilities
prob_L = Fs.fl_hat(rate_LH, rate_HL)
prob_H = Fs.fh_hat(rate_LH, rate_HL)

# Initialize arrays to store mutation counts and colony states
mutation_counts = np.zeros(num_colonies)
colony_states = np.zeros((num_colonies, 2))

# Initialize colonies
# colony_states[i, 0] = N_H, colony_states[i, 1] = N_L
for i in range(num_colonies):
    colony_states[i] = Fs.init_col(prob_H, N_0)

# Simulate over generations
for t in range(generations):
    for i in range(num_colonies):
        N_H, N_L = colony_states[i]
        colony_states[i] = Fs.next_state(N_H, N_L, rate_HL, rate_LH)
        mutation_counts[i] += Fs.m_tot(L, N_H, N_L, mu_H, mu_L)

# Calculate and print expected value and variance of mutations
expected_mutations = np.mean(mutation_counts)
variance_mutations = np.var(mutation_counts)
print(f'E(M) = {int(expected_mutations)}, V(M) = {int(variance_mutations)}')
print(f'V/E = {int(variance_mutations / expected_mutations)}')