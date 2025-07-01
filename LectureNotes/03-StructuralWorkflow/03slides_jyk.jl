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

# Simulate log-wages and compute actual log-wage
N = size(df,1)
β = [1.65,.4,.06,-.0002]
σ = .4;

df.exper = df.age .- ( 18*(1 .- df.collgrad) .+ 22*df.collgrad )
df.lwsim = β[1] .+ β[2]*df.collgrad .+ β[3]*df.exper .+ β[4]*df.exper.^2 .+ σ*randn(N)
df.lw    = log.(df.wage)

show(describe(df; cols=[:lw, :lwsim]), allrows=true)

# Estimate wage equation on employed sample
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
