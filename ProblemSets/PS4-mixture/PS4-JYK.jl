# Problem Set 4 - Question 1 (with PS3 starting values)
# Multinomial Logit with Automatic Differentiation
# Jae Yung Kim

# Load required packages
using Pkg
# Uncomment if packages are not installed:
# Pkg.add("Distributions")
# Pkg.add("ForwardDiff")

using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using ForwardDiff
using Distributions

# Set random seed for reproducibility
Random.seed!(1234)

# Load the data
println("Loading data...")
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Display column names
println("\nAvailable columns in the dataset:")
println(names(df))

# Prepare the data - Convert to Float64 for ForwardDiff compatibility
X = Float64.(hcat(df.age, df.white, df.collgrad))
Z = Float64.(hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
    df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8))
y = Int.(df.occ_code)

# Get dimensions
N = size(df, 1)
J = 8
K = size(X, 2)

# Calculate panel dimensions
n_individuals = length(unique(df.idcode))
n_years = length(unique(df.year))

println("\n" * repeat("=", 50))
println("DATA SUMMARY")
println(repeat("=", 50))
println("Number of individuals: ", n_individuals)
println("Number of years: ", n_years)
println("Total observations: ", N)
println("Average obs per individual: ", round(N / n_individuals, digits=2))
println("Number of choices: ", J)
println("Individual characteristics (X): ", K, " variables")
println("Alternative-specific vars (Z): ", size(Z, 2), " variables")

# Display occupation frequency
println("\nOccupation distribution:")
occupation_names = ["Professional/Technical", "Managers/Admin", "Sales",
    "Clerical/Unskilled", "Craftsmen", "Operatives",
    "Transport", "Other"]
for j in 1:J
    count = sum(y .== j)
    pct = 100 * count / N
    println("  $j. $(occupation_names[j]): $count ($(round(pct, digits=1))%)")
end

# Modified compute_probs function for ForwardDiff compatibility
function compute_probs(β, γ, X, Z, J)
    N = size(X, 1)
    K = size(X, 2)

    # Use similar to preserve type for ForwardDiff
    V = similar(β, N, J)
    V .= zero(eltype(β))

    # Compute utilities for alternatives 1 to J-1
    for j in 1:(J-1)
        for i in 1:N
            V[i, j] = dot(X[i, :], β[:, j]) + γ * (Z[i, j] - Z[i, J])
        end
    end
    # V[:, J] remains 0 (base category)

    # Compute probabilities
    exp_V = exp.(V)
    probs = exp_V ./ sum(exp_V, dims=2)

    return probs
end

# Modified log-likelihood function
function loglik(θ, X, Z, y, J)
    N = size(X, 1)
    K = size(X, 2)

    # Extract parameters
    β = reshape(θ[1:(K*(J-1))], K, J - 1)
    γ = θ[end]

    # Compute probabilities
    probs = compute_probs(β, γ, X, Z, J)

    # Compute negative log-likelihood
    ll = zero(eltype(θ))
    for i in 1:N
        ll -= log(probs[i, y[i]] + 1e-10)
    end

    return ll
end

# Get starting values FROM PS3
println("\n" * repeat("=", 50))
println("INITIALIZATION WITH PS3 STARTING VALUES")
println(repeat("=", 50))

# PS3 starting values
startvals_ps3 = [0.0403744; 0.2439942; -1.57132; 0.0433254; 0.1468556; -2.959103;
    0.1020574; 0.7473086; -4.12005; 0.0375628; 0.6884899; -3.65577;
    0.0204543; -0.3584007; -4.376929; 0.1074636; -0.5263738; -6.199197;
    0.1168824; -0.2870554; -5.322248; 1.307477]

# Reshape β parameters (first 21 values: 3×7 matrix)
β_start = reshape(startvals_ps3[1:21], K, J - 1)
γ_start = startvals_ps3[22]

println("Starting values from PS3:")
println("β matrix (", K, "×", J - 1, "):")
coef_names = ["age", "white", "collgrad"]
choice_names = occupation_names[1:7]
for j in 1:(J-1)
    println("\n  $(choice_names[j]):")
    for i in 1:K
        println("    $(coef_names[i]): $(round(β_start[i,j], digits=6))")
    end
end
println("\nγ (wage coefficient): $(round(γ_start, digits=6))")

θ_start = startvals_ps3  # Use the full vector directly
println("\nNumber of parameters to estimate: ", length(θ_start))
println("  β parameters (K × (J-1)): ", K, " × ", J - 1, " = ", K * (J - 1))
println("  γ parameter: 1")
println("  Total: ", length(θ_start))

# Test objective function
initial_ll = loglik(θ_start, X, Z, y, J)
println("\nInitial negative log-likelihood with PS3 values: ", round(initial_ll, digits=4))

# Optimization
println("\n" * repeat("=", 50))
println("OPTIMIZATION")
println(repeat("=", 50))
println("Starting optimization with automatic differentiation...")
println("Using PS3 starting values should speed up convergence...")
flush(stdout)

# Create objective function
obj = θ -> loglik(θ, X, Z, y, J)

# Run optimization with automatic differentiation
start_time = time()
result = optimize(
    obj,
    θ_start,
    LBFGS(),
    Optim.Options(
        show_trace=true,
        iterations=1000,
        g_tol=1e-5,
        show_every=10
    );
    autodiff=:forward
)
elapsed_time = time() - start_time

# Extract results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:(K*(J-1))], K, J - 1)
γ_hat = θ_hat[end]

# Display results
println("\n" * repeat("=", 50))
println("ESTIMATION RESULTS")
println(repeat("=", 50))
println("Converged: ", Optim.converged(result))
println("Iterations: ", Optim.iterations(result))
println("Time elapsed: ", round(elapsed_time / 60, digits=2), " minutes")
println("Final log-likelihood: ", round(-Optim.minimum(result), digits=4))

# Parameter estimates
println("\n" * repeat("=", 50))
println("PARAMETER ESTIMATES")
println(repeat("=", 50))

println("\nβ coefficients (vs. Other):")
println(repeat("-", 40))
for j in 1:(J-1)
    println("\n$(choice_names[j]):")
    for i in 1:K
        println("  $(coef_names[i]): $(round(β_hat[i,j], digits=4))")
    end
end

println("\nγ (wage coefficient): $(round(γ_hat, digits=4))")

# Compare with PS3 starting values
println("\n" * repeat("=", 50))
println("COMPARISON: PS3 Starting Values vs PS4 Estimates")
println(repeat("=", 50))
for j in 1:(J-1)
    println("\n$(choice_names[j]):")
    for i in 1:K
        change = β_hat[i, j] - β_start[i, j]
        pct_change = abs(β_start[i, j]) > 0.001 ? 100 * change / β_start[i, j] : 0
        println("  $(coef_names[i]): $(round(β_start[i,j], digits=4)) → $(round(β_hat[i,j], digits=4)) (change: $(round(change, digits=4)))")
    end
end
println("\nγ: $(round(γ_start, digits=4)) → $(round(γ_hat, digits=4)) (change: $(round(γ_hat - γ_start, digits=4)))")

# Compute standard errors
println("\n" * repeat("=", 50))
println("STANDARD ERRORS")
println(repeat("=", 50))
println("Computing Hessian for standard errors...")

try
    H = ForwardDiff.hessian(obj, θ_hat)
    vcov = inv(H)
    se = sqrt.(diag(vcov))

    se_β = reshape(se[1:(K*(J-1))], K, J - 1)
    se_γ = se[end]

    # Display results with standard errors
    println("\nCoefficients with standard errors:")
    println(repeat("-", 40))
    for j in 1:(J-1)
        println("\n$(choice_names[j]):")
        for i in 1:K
            coef = β_hat[i, j]
            stderr = se_β[i, j]
            tstat = coef / stderr
            sig = abs(tstat) > 2.576 ? "***" : abs(tstat) > 1.96 ? "**" : abs(tstat) > 1.645 ? "*" : ""
            println("  $(coef_names[i]): $(round(coef, digits=4)) ($(round(stderr, digits=4))) $sig")
        end
    end

    tstat_γ = γ_hat / se_γ
    sig_γ = abs(tstat_γ) > 2.576 ? "***" : abs(tstat_γ) > 1.96 ? "**" : abs(tstat_γ) > 1.645 ? "*" : ""
    println("\nγ: $(round(γ_hat, digits=4)) ($(round(se_γ, digits=4))) $sig_γ")
    println("\nSignificance: *** p<0.01, ** p<0.05, * p<0.10")

catch e
    println("Warning: Could not compute standard errors")
    println("Error: ", e)
end

# Model fit statistics
println("\n" * repeat("=", 50))
println("MODEL FIT")
println(repeat("=", 50))

probs_hat = compute_probs(β_hat, γ_hat, X, Z, J)
avg_probs = vec(mean(probs_hat, dims=1))
actual_freq = [sum(y .== j) / N for j in 1:J]

println("\nActual vs Predicted Probabilities:")
println(repeat("-", 40))
println("Choice | Actual | Predicted | Diff")
println(repeat("-", 40))
for j in 1:J
    name = occupation_names[j]
    diff = avg_probs[j] - actual_freq[j]
    println("$j. $(rpad(name[1:min(12, length(name))], 12)) | $(round(actual_freq[j], digits=3)) | $(round(avg_probs[j], digits=3)) | $(round(diff, digits=3))")
end

# Pseudo R-squared
ll_model = -Optim.minimum(result)
ll_null = N * log(1 / J)
pseudo_r2 = 1 - (ll_model / ll_null)
println("\nPseudo R²: $(round(pseudo_r2, digits=4))")

println("\n" * repeat("=", 50))
println("QUESTION 1 COMPLETE!")
println(repeat("=", 50))

# Save results
results_q1 = Dict(
    "theta_hat" => θ_hat,
    "beta_hat" => β_hat,
    "gamma_hat" => γ_hat,
    "log_likelihood" => ll_model,
    "ps3_startvals" => startvals_ps3
)
println("\nResults saved in 'results_q1' dictionary")

####################################################
####################################################
####################################################
####################################################

# Problem Set 4 - Question 3

println("\n" * repeat("=", 50))
println("QUESTION 3 (a): GAUSSIAN QUADRATURE INTEGRATION CHECK")
println(repeat("=", 50))

using Distributions
include("lgwt.jl") # make sure lgwt.jl is in the working folder

# Define the standard normal N(0,1)
d = Normal(0, 1)

# Choose number of quadrature points K and bounds [a,b]
# For a Normal, ±4σ covers ~99.9937% of mass, so it's a great finite window.
K = 7
a, b = -4.0, 4.0

# Get nodes (ξ_r) and weights (ω_r) on [a,b]
nodes, weights = lgwt(K, a, b)

# 1) Check that ∫ φ(x) dx ≈ 1 over [-4,4]
# (It won't be exactly 1 because we truncate tiny tails beyond ±4.)
approx_mass = sum(weights .* pdf.(d, nodes))
println("Approximate ∫ pdf over [-4,4]: ", approx_mass)   # ~0.99994

# 2) Check that ∫ x φ(x) dx ≈ μ = 0
approx_mean = sum(weights .* nodes .* pdf.(d, nodes))
println("Approximate E[X] over [-4,4]: ", approx_mean)    # ~0
