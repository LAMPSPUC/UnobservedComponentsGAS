using Random
using Distributions
using Plots
using CSV, DataFrames, Dates


function sample_variances(μ, σ)

    σ2ϵ = rand(LogNormal(μ, σ))
    σ2η = rand(LogNormal(μ, σ))
    σ2ζ = rand(LogNormal(μ, σ))
    σ2ω = rand(LogNormal(μ, σ))

    return Dict("irregular" => σ2ϵ, 
                "level" => σ2η, 
                "slope" => σ2ζ, 
                "seasonality" => σ2ω)
end

function sample_initial_values(μ, σ)
    μ1 = rand(Normal(μ, σ)) 
    β1 = rand(Normal(μ, σ))
    γ1 = rand(Normal(μ, σ), 6)

    Dict("level" => μ1, 
          "slope" => β1, 
          "seasonality" => γ1)
end


function simulate_series(T, variances, initial_values)
    s = 12  # Length of the seasonal cycle
    K = 6  # Number of harmonics
    σ2ϵ = variances["irregular"] # Variance of observation noise
    σ2η = variances["level"]  # Variance of level noise
    σ2ζ = variances["slope"]
    σ2ω = variances["seasonality"]  # Variance of seasonal noise

    # Initialize arrays to store the series
    y = zeros(T)
    μ = zeros(T + 1)
    β = zeros(T + 1)
    γ = zeros(T + 1, K)
    γ_star = zeros(T + 1, K)

    ω = rand(Normal(0, sqrt(σ2ω)), T, K)
    ω_star = rand(Normal(0, sqrt(σ2ω)), T, K)

    # Initial state
    μ[1] = initial_values["level"]
    β[1] = initial_values["slope"]
    γ[1, :] = initial_values["seasonality"]
    λ = [2 * π * j / s for j in 1:K]

    # Simulate the series
    for t in 1:T   
        γ_t = sum([γ[t, j] for j in 1:K])
        y[t] = μ[t] + γ_t + rand(Normal(0, sqrt(σ2ϵ)))
        #μ[t + 1] = μ[t] + rand(Normal(0, sqrt(σ2η)))
        μ[t + 1] = μ[t] + β[t] + rand(Normal(0, sqrt(σ2η)))
        β[t + 1] = β[t] + rand(Normal(0, sqrt(σ2ζ)))
        if t < T
            for j in 1:K
                γ[t + 1, j] = cos(λ[j]) * γ[t, j] + sin(λ[j]) * γ_star[t, j] + ω[t, j]
                γ_star[t + 1, j] = -sin(λ[j]) * γ[t, j] + cos(λ[j]) * γ_star[t, j] + ω_star[t, j]
            end
        end
    end
    return y[s+1:end]
end


T = 200
N = 100
Y = zeros(N, T-12)
for n in 1:N
# Random.seed!(1234)
    variances = sample_variances(1, 10)
    initial_values = sample_initial_values(0, 100)
    y = simulate_series(T, variances, initial_values)
    Y[n, :] .= y
end


function train_test_split(Y, steps_ahead)
    return Y[:, 1:end-steps_ahead], Y[:, end-steps_ahead+1:end]
end

Y_train, Y_test = train_test_split(Y, 18)

CSV.write("SimulacaoMEB_Train.csv",DataFrame(Y_train, :auto))
CSV.write("SimulacaoMEB_Test.csv",DataFrame(Y_test, :auto))
