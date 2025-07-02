using CSV, DataFrames, Statistics, GLM, PrettyTables, Plots

println("Current working directory: $(pwd())")
cd(dirname(@__FILE__))
println("Changed to script directory: $(pwd())")

# --- Load and inspect data
df = CSV.read("Data/slides3data.csv", DataFrame; missingstrings=["NA"])
size(df)
pretty_table(first(df, 5); display_size = (-1, -1))
show(describe(df), allrows=true)

# Compute potential experience
df.exper = df.age .- ( 18*(1 .- df.collgrad) .+ 22*df.collgrad )

# Simulate log-wages and compute actual log-wage
N = size(df,1)
β = [1.65,.4,.06,-.0002]
σ = .4;

df.lwsim = β[1] .+ β[2]*df.collgrad .+ β[3]*df.exper .+ β[4]*df.exper.^2 .+ σ*randn(N)
df.lw    = log.(df.wage)

show(describe(df; cols=[:lw, :lwsim]), allrows=true)

## Simulated policy function
using Plots

# Parameters for schooling utility
α0 = 0.5     # intercept
α1 = 1.0     # parent_college effect
α2 = -0.3    # effect of EFC (cost)

# Range of EFC (Expected Family Contribution) values
efc_vals = 0:0.1:10

# Policy function (probability of college) for parent_college = 0
p_school_z0 = @. 1 / (1 + exp(-(α0 + α1 * 0 + α2 * efc_vals)))

# Policy function for parent_college = 1
p_school_z1 = @. 1 / (1 + exp(-(α0 + α1 * 1 + α2 * efc_vals)))

# Plotting
plot(efc_vals, p_school_z0, label="Parent College = 0", linewidth=2)
plot!(efc_vals, p_school_z1, label="Parent College = 1", linewidth=2)
xlabel!("EFC (Expected Family Contribution)")
ylabel!("Pr(Go to College)")
title!("Simulated Schooling Policy Function")

## Numerical Comparative Statics

# Fixed range of EFC
efc_vals = 0:0.1:10

# Parameters to test: α2 sensitivity to cost
α0 = 0.5
α1 = 1.0
α2_vals = [-0.1, -0.3, -0.5]  # Try low, medium, and high sensitivity to EFC

# Plot effect of changing α2 (EFC sensitivity) on schooling prob when parent_college = 0
p_funcs = [@. 1 / (1 + exp(-(α0 + α1 * 0 + α2 * efc_vals))) for α2 in α2_vals]

plot()
for (i, pf) in enumerate(p_funcs)
    plot!(efc_vals, pf, label="α₂ = $(α2_vals[i])", linewidth=2)
end
xlabel!("EFC (Expected Family Contribution)")
ylabel!("Pr(Go to College)")
title!("Comparative Statics: Effect of EFC Sensitivity (α₂)")


## Estimate wage equation on employed sample
β̂ = lm(@formula(lw ~ collgrad + exper + exper^2), df[df.employed.==1,:])
df.elwage = predict(β̂, df) # generates expected log wage for all observations
println("R² of wage regression: ", r2(β̂))                               # reports R2
println("RMSE: ", sqrt(deviance(β̂)/dof_residual(β̂)))  # reports root mean squared error

# Estimate schooling choice logit
α̂ = glm(@formula(collgrad ~ parent_college + efc), df, Binomial(), LogitLink())

# Estimate employment choice logit
γ̂ = glm(@formula(employed ~ elwage + numkids), df, Binomial(), LogitLink())

# --- Counterfactual: reduce EFC by $1,000
df_cfl     = deepcopy(df)
df_cfl.efc = df.efc .- 1         # change value of efc to be $1,000 less

df.basesch = predict(α̂, df)     # generates expected log wage for all observations
df.cflsch  = predict(α̂, df_cfl) # generates expected log wage for all observations

show(describe(df; cols=[:basesch, :cflsch]), allrows=true)

# ============================================================================================
# Visualization Section
# ============================================================================================

# 1. Distribution of Actual vs Simulated Log Wages
p1 = histogram(df.lw,
    bins=30,
    alpha=0.6,
    label="Actual log(wage)",
    xlabel="Log Wage",
    ylabel="Frequency",
    title="Distribution of Log Wages")
histogram!(df.lwsim,
    bins=30,
    alpha=0.6,
    label="Simulated log(wage)")

display(p1)
# savefig(p1, "dist_log_wage.png")

# 2. Pr(college) vs EFC (Expected Family Contribution)
p2 = scatter(df.efc, df.basesch,
    markersize=4,
    alpha=0.7,
    label="Baseline",
    xlabel="EFC (thousands of \$)",
    ylabel="Pr(collgrad)",
    title="Predicted College Graduation Probability vs EFC")
scatter!(df.efc, df.cflsch,
    markersize=4,
    alpha=0.7,
    label="–\$1k EFC",
    color=:orange)

display(p2)
# savefig(p2, "pr_college_vs_efc.png")
