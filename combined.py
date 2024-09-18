import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Parameters
num_experiments = 4000  # Number of experiments to simulate
num_generations = 20     # Total number of generations

# Mutation rates
mu_1 = 1e-6              # Mutation rate for non-mutator cells
mu_2 = 1e-4              # Mutation rate for mutator cells (mu_2 >> mu_1)

# Parameters for the switching process
r12 = 0.001  # Switching rate from non-mutator to mutator per generation
r21 = 0.1    # Switching rate from mutator to non-mutator per generation
f1_hat = r21 / (r12 + r21)  # Equilibrium fraction of non-mutator cells
f2_hat = r12 / (r12 + r21)  # Equilibrium fraction of mutator cells

# Adjusted mutation rate for the Luria-Delbrück process
mu_ld = mu_1 * f1_hat + mu_2 * f2_hat  # Adjusted mutation rate for Luria-Delbrück

# Starting population size
initial_population_size = 1  # Starting with 1000 individuals

def simulate_luria_delbruck_per_generation(mu_ld, num_generations, initial_population_size):
    """
    Simulate the standard Luria-Delbrück process per generation, counting mutation events.
    """
    num_mutation_events = 0
    num_non_mutants = initial_population_size  # Start with 1000 non-mutant cells
    for gen in range(num_generations):
        # Mutations occur during divisions
        divisions = num_non_mutants
        mutations = np.random.binomial(divisions, mu_ld)  # Count mutation events
        num_mutation_events += mutations  # Count total mutation events
        num_non_mutants = (num_non_mutants - mutations) * 2  # Non-mutant cells divide
    return num_mutation_events

def simulate_switching_process_per_generation(mu_1, mu_2, num_generations, r12, r21, initial_population_size):
    """
    Simulate the switching process per generation, counting mutation events.
    """
    # Initialize counts based on the equilibrium fractions
    initial_states = np.random.choice([1, 2], size=initial_population_size, p=[f1_hat, f2_hat])
    counts = {
        (1, False): np.sum(initial_states == 1),
        (2, False): np.sum(initial_states == 2),
        (1, True): 0,
        (2, True): 0
    }

    num_mutation_events = 0

    for gen in range(num_generations):
        new_counts = {
            (1, False): 0,
            (2, False): 0,
            (1, True): 0,
            (2, True): 0
        }
        for (state, mutant), num_cells in counts.items():
            if num_cells == 0:
                continue

            # State switching probabilities per generation
            if state == 1:
                prob_switch = r12
                states = [1, 2]
                probs = [1 - prob_switch, prob_switch]
            else:
                prob_switch = r21
                states = [2, 1]
                probs = [1 - prob_switch, prob_switch]

            # Determine the state of daughter cells after switching
            state_after_switch = np.random.choice(states, size=num_cells, p=probs)
            unique_states, counts_states = np.unique(state_after_switch, return_counts=True)

            for new_state, count in zip(unique_states, counts_states):
                # Count mutation events only for non-mutant cells
                if mutant:
                    new_counts[(new_state, True)] += count * 2
                else:
                    if new_state == 2:
                        # Mutations occur during divisions in mutator state
                        divisions = count
                        mutations = np.random.binomial(divisions, mu_2 * 2)
                        num_mutation_events += mutations  # Count total mutation events
                        non_mutant_offspring = (divisions * 2) - mutations
                        new_counts[(new_state, False)] += non_mutant_offspring
                        new_counts[(new_state, True)] += mutations
                    else:
                        # Mutations occur during divisions in non-mutator state
                        divisions = count
                        mutations = np.random.binomial(divisions, mu_1 * 2)
                        num_mutation_events += mutations  # Count total mutation events
                        non_mutant_offspring = (divisions * 2) - mutations
                        new_counts[(new_state, False)] += non_mutant_offspring
                        new_counts[(new_state, True)] += mutations

        counts = new_counts

    return num_mutation_events

# Run simulations
mutation_events_ld = []
mutation_events_switching = []

print("Simulating mutation events in standard Luria-Delbrück process per generation...")
for _ in tqdm(range(num_experiments)):
    mutations = simulate_luria_delbruck_per_generation(mu_ld, num_generations, initial_population_size)
    mutation_events_ld.append(mutations)

print("Simulating mutation events in switching process per generation...")
for _ in tqdm(range(num_experiments)):
    mutations = simulate_switching_process_per_generation(mu_1, mu_2, num_generations, r12, r21, initial_population_size)
    mutation_events_switching.append(mutations)

# Convert the data into a DataFrame for saving
df = pd.DataFrame({
    'Luria_Delbruck': mutation_events_ld,
    'Switching_Process': mutation_events_switching
})

# Get the current directory path
current_dir = os.getcwd()

# Save the data to a CSV file
csv_filename = os.path.join(current_dir, "mutation_events_simulation_data.csv")
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# Calculate variance and mean for both processes
mean_ld = np.mean(mutation_events_ld)
variance_ld = np.var(mutation_events_ld)
fano_factor_ld = variance_ld / mean_ld if mean_ld != 0 else 0

mean_switching = np.mean(mutation_events_switching)
variance_switching = np.var(mutation_events_switching)
fano_factor_switching = variance_switching / mean_switching if mean_switching != 0 else 0

# Print the results
print(f"Variance / Mean (Fano factor) for Luria-Delbrück (Mutation Events): {fano_factor_ld:.3f}")
print(f"Variance / Mean (Fano factor) for Switching Process (Mutation Events): {fano_factor_switching:.3f}")

# Plotting
max_mutations = max(max(mutation_events_ld), max(mutation_events_switching))
max_bin_edge = min(int(max_mutations), 10) + 1  # Adjust 10 to desired maximum bin edge

# Adjust bins: create bins from 0 to max_bin_edge, with a final bin for values >= max_bin_edge - 1
bins = list(range(max_bin_edge)) + [np.inf]

# Histogram data
hist_ld, _ = np.histogram(mutation_events_ld, bins=bins)
hist_switching, _ = np.histogram(mutation_events_switching, bins=bins)

# Positions for side-by-side bars
bar_width = 0.4
positions = np.arange(len(bins) - 1)
positions_ld = positions - bar_width / 2
positions_switching = positions + bar_width / 2

plt.figure(figsize=(12, 6))
plt.bar(positions_ld, hist_ld, width=bar_width, alpha=0.7,
        label='Luria-Delbrück (Mutation Events)', color='blue', edgecolor='black')
plt.bar(positions_switching, hist_switching, width=bar_width, alpha=0.7,
        label='Switching Process (Mutation Events)', color='red', edgecolor='black')

# Use log scale for the y-axis if mutation events span a large range
plt.yscale('log')

plt.xlabel('Number of Mutation Events')
plt.ylabel('Number of Experiments')
plt.title('Distribution of Mutation Events in Luria-Delbrück and Switching Processes')
plt.legend()
plt.grid(True)

# Use correct bin labels
labels = [str(i) for i in range(max_bin_edge - 1)] + [f'>= {max_bin_edge - 1}']
plt.xticks(positions, labels=labels)
plt.show()
