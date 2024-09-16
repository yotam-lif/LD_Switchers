import numpy as np
import matplotlib.pyplot as plt


def simulate_luria_delbruck(N, initial_cells, mutation_rate, final_time):
    """
    Simulate the Luria-Delbrück experiment with synchronized cell divisions.

    Parameters:
    N : int
        Number of cultures.
    initial_cells : int
        Initial number of cells in each culture.
    mutation_rate : float
        Mutation rate per cell division.
    growth_rate : float
        Growth rate of the cells.
    final_time : int
        Total time of the experiment (number of cell division cycles).

    Returns:
    list
        List containing the number of resistant bacteria in each culture.
    """
    results = []

    for _ in range(N):
        total_cells = initial_cells
        resistant_cells = 0

        for _ in range(final_time):
            # Each cell division cycle
            new_resistant_cells = int(np.random.exponential(mutation_rate * total_cells))
            resistant_cells += new_resistant_cells

            # Update the total number of cells
            total_cells *= 2

        results.append(resistant_cells)

    return results


# Parameters
N = 10 ** 5  # Number of cultures
initial_cells = 1  # Initial number of cells
mutation_rate = 1e-8  # Mutation rate per cell division
final_time = 30  # Total number of cell division cycles

# Simulate the experiment
resistant_counts = simulate_luria_delbruck(N, initial_cells, mutation_rate, final_time)

# Plot the results
plt.hist(resistant_counts, bins=100)
plt.xlabel('Number of Resistant Bacteria')
plt.ylabel('Frequency')
plt.title(f'gens = {final_time}, mu = {mutation_rate}')
plt.show()
