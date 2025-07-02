import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulate representative data
np.random.seed(0)
N = 1000
parent_college = np.random.binomial(1, 0.4, N)
efc = np.random.normal(5, 1.5, N)
α0, α1, α2 = -2, 1.2, -0.5  # coefficients from hypothetical α̂

# Compute schooling probability (logit)
log_odds = α0 + α1 * parent_college + α2 * efc
prob_school = 1 / (1 + np.exp(-log_odds))

# Create DataFrame for plotting
df_plot = pd.DataFrame({
    "ParentCollege": parent_college,
    "EFC": efc,
    "SchoolProb": prob_school
})

# Sort by EFC for smooth line
df_plot_sorted = df_plot.sort_values("EFC")

# Plot policy function: P(collgrad=1 | parent_college, efc)
plt.figure(figsize=(10, 6))
for pc_val in [0, 1]:
    subset = df_plot_sorted[df_plot_sorted["ParentCollege"] == pc_val]
    plt.plot(subset["EFC"], subset["SchoolProb"], label=f"Parent college = {pc_val}")

plt.xlabel("Expected Family Contribution (EFC)")
plt.ylabel("Probability of Completing College")
plt.title("Policy Function: Schooling Decision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
