import numpy as np
import matplotlib.pyplot as plt

# Parameters for standard case (converges)
delta_good = 0.95  # discount factor
beta = 1.0         # utility from consumption
T = 100            # number of value function iterations
grid_size = 100    # size of state space
states = np.linspace(1, 10, grid_size)
V_good = np.zeros(grid_size)
V_history_good = []

# Bellman iteration for good delta
for t in range(T):
    V_new = beta * np.log(states) + delta_good * V_good  # simple value update
    V_history_good.append(V_new.copy())
    if np.max(np.abs(V_new - V_good)) < 1e-6:
        break
    V_good = V_new

# Parameters for bad case (non-convergent)
delta_bad = 1.01  # discount factor too high
V_bad = np.zeros(grid_size)
V_history_bad = []

# Bellman iteration for bad delta
for t in range(T):
    V_new_bad = beta * np.log(states) + delta_bad * V_bad
    V_history_bad.append(V_new_bad.copy())
    if np.max(np.abs(V_new_bad - V_bad)) < 1e-6:
        break
    V_bad = V_new_bad

# Plotting value function evolution for a mid-state
mid_index = grid_size // 2
good_vals = [V[mid_index] for V in V_history_good]
bad_vals = [V[mid_index] for V in V_history_bad]

plt.figure()
plt.plot(good_vals, label="Converging (δ=0.95)")
plt.plot(bad_vals, label="Diverging (δ=1.01)")
plt.xlabel("Iteration")
plt.ylabel("Value at middle state")
plt.title("Value Function Convergence vs Divergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()