import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
num_experiments = 1000  # Number of experiments to simulate
num_generations = 16  # Total number of generations (adjust as needed)
mu = 1e-4  # Mutation rate per cell division

# Parameters for the switching process
r12 = 0.1  # Switching rate from non-mutator to mutator per generation
r21 = 0.05  # Switching rate from mutator to non-mutator per generation
f1_hat = r21 / (r12 + r21)  # Equilibrium fraction of non-mutator
f2_hat = r12 / (r12 + r21)  # Equilibrium fraction of mutator

def simulate_luria_delbruck_per_generation(mu, num_generations):
    """
    Simulate the standard Luria-Delbrück process per generation using counts.
    """
    num_mutants = 0
    num_non_mutants = 1  # Start with one non-mutant cell
    for gen in range(num_generations):
        # Mutations occur during divisions
        divisions = num_non_mutants
        mutations = np.random.binomial(divisions, mu)
        num_mutants = num_mutants * 2 + mutations  # Mutant cells double
        num_non_mutants = (num_non_mutants - mutations) * 2
    return num_mutants

def simulate_switching_process_per_generation(mu, num_generations, r12, r21):
    """
    Simulate the switching process per generation using counts.
    """
    # Cells are tracked by (state, mutant_status)
    # States: 1 = non-mutator, 2 = mutator
    # Mutant_status: False = non-mutant, True = mutant
    # Start with one cell in a randomly chosen initial state
    initial_state = np.random.choice([1, 2], p=[f1_hat, f2_hat])
    counts = {
        (1, False): 0,
        (2, False): 0,
        (1, True): 0,
        (2, True): 0
    }
    counts[(initial_state, False)] = 1

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
                # Mutant cells remain mutant
                if mutant:
                    # Mutant cells divide into mutant cells
                    new_counts[(new_state, True)] += count * 2
                else:
                    # Non-mutant cells may mutate if in mutator state
                    if new_state == 2:
                        # Mutations occur during divisions
                        divisions = count
                        mutations = np.random.binomial(divisions, mu * 2)
                        non_mutant_offspring = (divisions * 2) - mutations
                        new_counts[(new_state, False)] += non_mutant_offspring
                        new_counts[(new_state, True)] += mutations
                    else:
                        # Non-mutator cells divide without mutations
                        new_counts[(new_state, False)] += count * 2

        counts = new_counts

    num_mutants = sum(num_cells for (state, mutant), num_cells in counts.items() if mutant)
    return num_mutants

# Run simulations
mutant_counts_ld = []
mutant_counts_switching = []

print("Simulating standard Luria-Delbrück process per generation...")
for _ in tqdm(range(num_experiments)):
    mutants = simulate_luria_delbruck_per_generation(mu, num_generations)
    mutant_counts_ld.append(mutants)

print("Simulating switching process per generation...")
for _ in tqdm(range(num_experiments)):
    mutants = simulate_switching_process_per_generation(mu, num_generations, r12, r21)
    mutant_counts_switching.append(mutants)

# Plotting
max_mutants = max(max(mutant_counts_ld), max(mutant_counts_switching))
bins = np.arange(0, max_mutants + 2) - 0.5  # Bin edges

# Histogram data
hist_ld, _ = np.histogram(mutant_counts_ld, bins=bins)
hist_switching, _ = np.histogram(mutant_counts_switching, bins=bins)

# Positions for side-by-side bars
bar_width = 0.4
positions_ld = bins[:-1] - bar_width / 2
positions_switching = bins[:-1] + bar_width / 2

plt.figure(figsize=(12, 6))
plt.bar(positions_ld, hist_ld, width=bar_width, alpha=0.7, label='Standard Luria-Delbrück', color='blue', edgecolor='black')
plt.bar(positions_switching, hist_switching, width=bar_width, alpha=0.7, label='Switching Process', color='red', edgecolor='black')

plt.xlabel('Number of Mutants')
plt.ylabel('Number of Experiments')
plt.title('Distribution of Mutants in Luria-Delbrück and Switching Processes')
plt.legend()
plt.grid(True)
plt.xticks(bins[:-1], bins[:-1].astype(int))
plt.show()