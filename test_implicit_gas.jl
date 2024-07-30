using Distributions, JuMP, Ipopt, Plots, Random

T = 1000
σ = std(rand(T))
μ = rand(T)
μ[1] = 0
[μ[t+1] = μ[t] + 0.1*rand() for t = 1:T-1]

y = zeros(T)
[y[t] = rand(Normal(μ[t],σ)) for t = 1:T]
plot(y)
plot!(μ)

m = Model(Ipopt.Optimizer)
@variable(m, μ_update[1:T])
@variable(m, s[1:T])
@variable(m, ω)
@variable(m, ϕ)
μ_pred = 2*μ_update # inicializando com um expression qq
#@constraint(m, [t = 1:T], s[t] == μ_update[t] - μ_pred[t])
[μ_pred[t+1] = μ_update[t] for t = 1:T-1]
f(y,μ) = logpdf(Normal(μ,σ),y)
@operator(m,ℓ,2,f)
@objective(m, Max, sum(ℓ(y[t],μ_update[t]) + 0.5*s[t]^2 for t in 2:T))
optimize!(m)
plot(y)
plot!(value.(μ_update))
plot!(value.(μ_pred))
scatter(value.(μ̂),value.(μ))


# Modelo de nível local
import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/UnobservedComponentsGAS.jl")
using Distributions, JuMP, Ipopt, Plots, Random, CSV, DataFrames, LinearAlgebra


function implicit_gas(y::Vector{Float64}, λ::Float64)
    T = length(y)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 100.0)
    set_optimizer_attribute(model, "tol",  0.005) 

    @variable(model, m_update[1:T])
    @variable(model, σ  ≥ 0.0004)
    @variable(model, s[1:T])

    @variable(model, ω)
    @variable(model, ϕ)
 
    m_pred = Vector(undef, T)
    for t in 2:T
        m_pred[t] = ω + ϕ * m_update[t-1]
    end

    @expression(model, μ, m_update)

    @constraint(model,[t = 2:T], s[t] == m_update[t] - m_pred[t])

    f(y,μ, σ) = logpdf(Normal(μ,σ),y)
    @operator(model,ℓ,3,f)
    #@objective(model, Max, sum(ℓ(y[t],μ[t], σ) - λ*s[t]^2 for t in 2:T))
    @objective(model, Max, sum(ℓ(y[t],μ[t], σ) for t in 2:T) - λ*s'I(T)*s)
    
    set_start_value.(m_update, y)
    set_start_value(σ, std(diff(y)))

    optimize!(model)

    println(termination_status(model))
    return value.(m_update), value.(m_pred[2:end]), value(σ), value.(s), value(ϕ), value(ω)
    
end

#gerando séries
T = 1000
N = 10
series = zeros(T, 10)
Random.seed!(10)

for η in [0.1, 0.2, 0.5, 0.8]
    println("η = $η")
    for i in 1:N
        σ0 = std(rand(T))
        μ0 = rand(1)[1]
        μ = zeros(T)
        μ[1] = μ0

        for t = 2:T
            μ[t] = μ[t-1] + η * rand(Normal(0, 1))
        end
        y = zeros(T)
        [y[t] = rand(Normal(μ[t],σ0)) for t = 1:T]

        series[:, i] = y
    end

    dist = UnobservedComponentsGAS.NormalDistribution()

    model = UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, "random walk", "", missing)

    rmse_model = zeros(N)
    rmse_implicit_model = zeros(N)

    time_model = zeros(N)
    time_implicit_model = zeros(N)

    for i in 1:N
        println("serie $i")
        y = series[:, i]

        time_model[i]          = @elapsed fitted_model = UnobservedComponentsGAS.fit(model, y)
        time_implicit_model[i] = @elapsed μ_up, μ_pred, σ, s, ϕ, ω = implicit_gas(y, 0.5)

        println("ϕ = $ϕ")
        println("ω = $ω")
        rmse_model[i]          = sqrt(mean((y .-fitted_model.fit_in_sample).^2))
        rmse_implicit_model[i] = sqrt(mean((y[2:end] .- μ_pred).^2))

        plot(y[3:end], color = "black", linewidth = 1.5, label = "Simulated series")
        plot!(fitted_model.fit_in_sample[3:end], color = "red", linewidth = 1, alpha = 0.7, label = "Explicit model's fit")
        plot!(μ_pred[2:end], color = "blue", linewidth = 1,  alpha = 0.7,label = "Implicit model's fit")
        title!("Serie $i")
        savefig("results_figures/eta = $η/serie$(i).png")
    end

    results = hcat(collect(1:N), time_model, time_implicit_model, rmse_model, rmse_implicit_model)

    CSV.write("results_figures/eta = $η/results.csv", DataFrame(results, ["serie", "time ESD", "time ISD", "rmse ESD", "rmse ISD"]))
end
hcat(time_model, time_implicit_model)
hcat(rmse_model, rmse_implicit_model)

value(ω)

plot(y[3:end])
plot!(value.(μ)[3:end])
plot!(value.(m_pred[3:end]))