import numpy as np
import matplotlib.pyplot as plt


def simulate_luria_delbruck(N, initial_cells, mutation_rate, growth_rate, final_time):
    """
    Simulate the Luria-Delbrück experiment.

    Parameters:
    N : int
        Number of cultures.
    initial_cells : int
        Initial number of cells in each culture.
    mutation_rate : float
        Mutation rate per cell division.
    growth_rate : float
        Growth rate of the cells.
    final_time : float
        Total time of the experiment.

    Returns:
    list
        List containing the number of resistant bacteria in each culture.
    """
    results = []

    for _ in range(N):
        total_cells = initial_cells
        resistant_cells = 0

        # Event-driven simulation
        time = 0
        while time < final_time:
            # Time to next cell division
            division_time = np.random.exponential(1 / (growth_rate * total_cells))
            time += division_time

            if time >= final_time:
                break

            # Mutation event
            if np.random.rand() < mutation_rate:
                resistant_cells += 1

            # Update the total number of cells
            total_cells += 1

        results.append(resistant_cells)

    return results


# Parameters
N = 1000  # Number of cultures
initial_cells = 10  # Initial number of cells
mutation_rate = 1e-8  # Mutation rate per cell division
growth_rate = 1  # Growth rate per cell
final_time = 10  # Total time of the experiment

# Simulate the experiment
resistant_counts = simulate_luria_delbruck(N, initial_cells, mutation_rate, growth_rate, final_time)

# Plot the results
plt.hist(resistant_counts, bins=50, edgecolor='black')
plt.xlabel('Number of Resistant Bacteria')
plt.ylabel('Frequency')
plt.title('Distribution of Resistant Bacteria in Luria-Delbrück Experiment')
plt.show()
