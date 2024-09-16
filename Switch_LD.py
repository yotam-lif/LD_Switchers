import numpy as np
import matplotlib.pyplot as plt


def simulate_LD_switch(N, initial_cells, mutation_rate, r_12, r_21, final_time):
    """
    Simulate the Luria-Delbrück experiment with synchronized cell divisions and stochastic switching between cell states.

    Parameters:
    N : int
        Number of cultures.
    initial_cells : int
        Initial number of cells in each culture.
    mutation_rate : float
        Mutation rate per cell division for mutator cells.
    r_12 : float
        Rate of switching from non-mutator to mutator.
    r_21 : float
        Rate of switching from mutator to non-mutator.
    final_time : int
        Total time of the experiment (number of cell division cycles).

    Returns:
    list
        List containing the number of resistant bacteria in each culture.
    """
    results = []

    p_mut = r_12 / (r_12 + r_21)

    for _ in range(N):
        non_mutator_cells = initial_cells
        mutator_cells = 0
        resistant_cells = 0

        # Initialize the state of the originator cell
        if np.random.rand() < p_mut:
            mutator_cells = initial_cells
            non_mutator_cells = 0

        for _ in range(final_time):
            # Mutator cells can mutate
            new_resistant_cells = int(np.random.exponential(mutation_rate * mutator_cells))
            resistant_cells += new_resistant_cells

            # Calculate new cells switching states
            new_mutators = np.random.binomial(non_mutator_cells, r_12)
            new_non_mutators = np.random.binomial(mutator_cells, r_21)

            # Update cell counts
            non_mutator_cells = 2 * non_mutator_cells - new_mutators + new_non_mutators
            mutator_cells = 2 * mutator_cells + new_mutators - new_non_mutators

        results.append(resistant_cells)

    return results


# Parameters
N = 10 ** 5  # Number of cultures
initial_cells = 1   # Initial number of cells (1 originator per colony)
mutation_rate = 1e-8  # Mutation rate per cell division for mutator cells
r_12 = 0.01  # Rate of switching from non-mutator to mutator
r_21 = 0.001  # Rate of switching from mutator to non-mutator
final_time = 30  # Total number of cell division cycles

# Simulate the experiment
resistant_counts = simulate_LD_switch(N, initial_cells, mutation_rate, r_12, r_21, final_time)

# Plot the results
plt.hist(resistant_counts, bins=100)
plt.xlabel('Number of Resistant Bacteria')
plt.ylabel('Frequency')
plt.title(f'r_12 = {r_12}, r_21 = {r_21}, gens = {final_time}, mu = {mutation_rate}')
plt.show()
