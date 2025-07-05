# Problem Set 2 - ECON 6343: Econometrics III

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Problem 1: Basic optimization in Julia

using Optim

# Define the function f(x) = -x^4 - 10x^3 - 2x^2 - 3x - 2
f(x) = -x[1]^4 - 10*x[1]^3 - 2*x[1]^2 - 3*x[1] - 2

# Since Optim minimizes, we need to minimize -f(x) to find the maximum of f(x)
negf(x) = x[1]^4 + 10*x[1]^3 + 2*x[1]^2 + 3*x[1] + 2

# Set a random starting value
startval = rand(1)  # random number as starting value

# Optimize using LBFGS algorithm
result = optimize(negf, startval, LBFGS())

# Display the results
println("Optimization Results:")
println("Full result:")
println(result)
println()

println("Minimum value (negative of maximum): ", result.minimum)
println("Maximum value of f(x): ", -result.minimum)
println("Optimizer (argmax): ", result.minimizer)
println("x* ≈ ", round(result.minimizer[1], digits=6))

# Verify by evaluating f(x) at the optimizer - calculate it inline
println("Verification - f(x*) = ", f(result.minimizer))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Problem 2: OLS estimation using optimization

using DataFrames
using CSV
using HTTP
using LinearAlgebra  # for matrix operations
using GLM

# Import and set up the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Create design matrix X and dependent variable y
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

# Define the OLS objective function (sum of squared residuals)
# married_i = β₀ + β₁×age_i + β₂×1[race_i=1] + β₃×1[collgrad_i=1] + u_i
function ols(beta, X, y)
    ssr = (y - X*beta)'*(y - X*beta)
    return ssr[1]  # return scalar value
end

# Use Optim to minimize the sum of squared residuals
println("\n Problem #2\n")
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),
                       Optim.Options(g_tol=1e-6, iterations=100_000,
                                   show_trace=true))

println("\nOLS Estimates from Optim:")
println("β̂ = ", beta_hat_ols.minimizer)
println()

# Check our answer using the analytical OLS formula
bols = inv(X'*X)*X'*y
println("OLS Estimates from analytical formula:")
println("β̂ = ", bols)
println()

# Check using GLM package
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("OLS Estimates from GLM package:")
println(bols_lm)

# Problem 3: Logit estimation using optimization

using Optim
using GLM

# We'll use the same data from Problem 2
# X and y should already be defined from Problem 2
# If not, recreate them:
# X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
# y = df.married.==1

# Define the logit likelihood function
function logit_likelihood(beta, X, y)
    # Calculate linear prediction: X*beta
    xb = X * beta
    
    # Calculate probabilities using logistic function: exp(xb)/(1 + exp(xb))
    # Use logistic function: 1/(1 + exp(-xb)) to avoid overflow
    probs = 1 ./ (1 .+ exp.(-xb))
    
    # Calculate log-likelihood
    # LL = Σ[y*log(p) + (1-y)*log(1-p)]
    ll = sum(y .* log.(probs) .+ (1 .- y) .* log.(1 .- probs))
    
    return ll
end

# Define negative log-likelihood (since Optim minimizes)
function neg_logit_likelihood(beta, X, y)
    return -logit_likelihood(beta, X, y)
end

# Estimate logit model using Optim
println("Estimating Logit Model using Optim...")
logit_result = optimize(b -> neg_logit_likelihood(b, X, y), 
                       rand(size(X,2)), 
                       LBFGS(),
                       Optim.Options(g_tol=1e-6, iterations=100_000,
                                   show_trace=true))

println("Logit Estimates from Optim:")
println("β̂ = ", logit_result.minimizer)
println("Log-likelihood = ", -logit_result.minimum)
println()

# Check our answer using GLM package
println("Logit Estimates from GLM package:")
df.white = df.race.==1  # Make sure this is defined
logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Coefficients: ", coef(logit_glm))
println("Log-likelihood: ", loglikelihood(logit_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Problem 3: Logit estimation using optimization

using Optim
using GLM

# We'll use the same data from Problem 2
# X and y should already be defined from Problem 2
# If not, recreate them:
# X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
# y = df.married.==1

# Define the logit likelihood function
function logit_likelihood(beta, X, y)
    # Calculate linear prediction: X*beta
    xb = X * beta
    
    # Calculate probabilities using logistic function: exp(xb)/(1 + exp(xb))
    # Use logistic function: 1/(1 + exp(-xb)) to avoid overflow
    probs = 1 ./ (1 .+ exp.(-xb))
    
    # Calculate log-likelihood
    # LL = Σ[y*log(p) + (1-y)*log(1-p)]
    ll = sum(y .* log.(probs) .+ (1 .- y) .* log.(1 .- probs))
    
    return ll
end

# Define negative log-likelihood (since Optim minimizes)
function neg_logit_likelihood(beta, X, y)
    return -logit_likelihood(beta, X, y)
end

# Estimate logit model using Optim
println("\n Problem #3\n")
println("Estimating Logit Model using Optim...")
logit_result = optimize(b -> neg_logit_likelihood(b, X, y), 
                       rand(size(X,2)), 
                       LBFGS(),
                       Optim.Options(g_tol=1e-6, iterations=100_000,
                                   show_trace=true))

println("\nLogit Estimates from Optim:")
println("β̂ = ", logit_result.minimizer)
println("Log-likelihood = ", -logit_result.minimum)
println()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Problem 4: Logit estimation using GLM package

# Check our answer using GLM package
println("\n Problem #4\n")
println("Logit Estimates from GLM package:")
df.white = df.race.==1  # Make sure this is defined
logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Coefficients: ", coef(logit_glm))
println("Log-likelihood: ", loglikelihood(logit_glm))

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Problem 5: Multinomial Logit estimation using optimization

# Problem 5: Multinomial Logit estimation using optimization

using Optim, DataFrames, CSV, HTTP, FreqTables, LinearAlgebra, Statistics

# 1. Load & clean
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

println("Original occupation frequencies:")
freqtable(df, :occupation)

# drop missing occupations
df = dropmissing(df, :occupation)

# collapse 8–13 into 7
for j in 8:13
    df[df.occupation .== j, :occupation] .= 7
end

println("\nCleaned occupation frequencies:")
freqtable(df, :occupation)

# 2. Build X and y
X = [ones(size(df,1)) df.age df.race .== 1 df.collgrad .== 1]  # (N×K), K=4
y = df.occupation                                            # values in 1:7

N, K = size(X)
J = 7                   # number of alternatives
M = J - 1               # number of free choice equations

println("\nModel setup:")
println("N (observations): $N")
println("K (covariates): $K") 
println("J (choices): $J")
println("M (estimated choice equations): $M")
println("Total parameters: $(K*M)")

# 3. Negative log-likelihood
function neglogit_mnl(flatβ)
    B = reshape(flatβ, K, M)      # 4×6 matrix
    ll = 0.0
    for i in 1:N
        η    = B' * X[i, :]       # (6×4)*(4×1) ⇒ 6×1 vector of utilities
        expη = exp.(η)
        denom = 1 + sum(expη)
        yi   = y[i]
        if yi < J
            ll += log(expη[yi] / denom)
        else
            ll += log(1 / denom)
        end
    end
    return -ll
end

# 4. Try different starting values as specified in problem
starting_values = [
    ("zeros", zeros(K * M)),
    ("U[0,1]", rand(K * M)),
    ("U[-1,1]", 2*rand(K * M) .- 1)
]

best_result = nothing
best_ll = Inf

for (name, initβ) in starting_values
    println("\n--- Trying starting values: $name ---")
    
    try
        res = optimize(
            neglogit_mnl,
            initβ,
            LBFGS(),
            Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100)
        )
        
        println("Converged: $(Optim.converged(res))")
        println("Objective value: $(res.minimum)")
        println("Log-likelihood: $(-res.minimum)")
        
        if res.minimum < best_ll
            best_ll = res.minimum
            best_result = res
            println("*** New best result! ***")
        end
        
    catch e
        println("Failed with error: $e")
    end
end

# 5. Extract and display best results
if best_result !== nothing
    β_hat = reshape(best_result.minimizer, K, M)
    
    println("\n" * "="^60)
    println("FINAL MULTINOMIAL LOGIT RESULTS")
    println("="^60)
    println("Converged: $(Optim.converged(best_result))")
    println("Log-likelihood: $(-best_result.minimum)")
    
    println("\n--- Multinomial Logit Coefficients (vs base = 7) ---")
    var_names = ["Intercept", "Age", "White", "College Grad"]
    
    for k in 1:K
        print(rpad(var_names[k], 12))
        for j in 1:M
            print(rpad(string(round(β_hat[k, j], digits=4)), 10))
        end
        println()
    end
    
    println("\nChoice equations (columns): 1, 2, 3, 4, 5, 6 vs base 7")
else
    println("All optimization attempts failed!")
end