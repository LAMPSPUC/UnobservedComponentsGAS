using Distributions, JuMP, Ipopt, Plots, Random

# Modelo de nível local
import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/UnobservedComponentsGAS.jl")
using Distributions, JuMP, Ipopt, Plots, Random, CSV, DataFrames, LinearAlgebra

function simulate_local_linear_trend(T, N, σ²_μ, σ²_β, σ²_ε)
    # Inicialização das séries
    y = zeros(T, N)
    
    # Inicialização dos estados
    μ = zeros(T, N)
    β = zeros(T, N)
    
    # Distribuições dos ruídos
    d_ημ = Normal(0, sqrt(σ²_μ))
    d_ηβ = Normal(0, sqrt(σ²_β))
    d_ε = Normal(0, sqrt(σ²_ε))
    
    # Condições iniciais
    μ[1, :] = randn(N)
    β[1, :] = randn(N)
    
    for n in 1:N
        for t in 2:T
            # Atualização dos estados
            μ[t, n] = μ[t-1, n] + β[t-1, n] + rand(d_ημ)
           β[t, n] = β[t-1, n] + rand(d_ηβ)
            
            # Geração da observação
            y[t, n] = μ[t, n] + rand(d_ε)
        end
    end
    
    return y
end

function simulate_local_level(T, N, σ²_μ, σ²_ε)
    # Inicialização das séries
    y = zeros(T, N)
    
    # Inicialização dos estados
    μ = zeros(T, N)
    
    # Distribuições dos ruídos
    d_ημ = Normal(0, sqrt(σ²_μ))
    d_ε = Normal(0, sqrt(σ²_ε))
    
    # Condições iniciais
    μ[1, :] = randn(N)
    
    for n in 1:N
        for t in 2:T
            # Atualização dos estados
            μ[t, n] = μ[t-1, n]  + rand(d_ημ)
            
            # Geração da observação
            y[t, n] = μ[t, n] + rand(d_ε)
        end
    end
    
    return y
end

function implicit_local_level_gas(y::Vector{Float64}, λ::Float64)
    T = length(y)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 100.0)
    set_optimizer_attribute(model, "tol",  0.005) 

    @variable(model, m_update[1:T])
    @variable(model, σ  ≥ 0.0004)
    @variable(model, s[1:T])

    @variable(model, -2 ≤ κ ≤ 2)

    #passo de previsão
    m_pred = Vector(undef, T)
    for t in 2:T
        m_pred[t] = m_update[t-1]  +κ * s[t-1]
    end

    @expression(model, μ, m_update)

    @constraint(model,[t = 2:T], s[t] == m_update[t] - m_pred[t])
   
    f(y,μ, σ) = logpdf(Normal(μ,σ),y)
    @operator(model,ℓ,3,f)
    @objective(model, Max, sum(ℓ.(y,μ, σ)) - λ*s'I(T)*s)
    
    set_start_value.(m_update, y)
    set_start_value(σ, std(diff(y)))

    optimize!(model)

    println(termination_status(model))
    return value.(m_update), value.(m_pred[2:end]), value(σ), value.(s), value(κ)
    
end

function implicit_local_linear_trend_gas(y::Vector{Float64}, λ::Float64)
    T = length(y)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 100.0)
    set_optimizer_attribute(model, "tol",  0.005) 

    @variable(model, m_update[1:T])
    @variable(model, β_update[1:T])
    @variable(model, σ  ≥ 0.0004)
    @variable(model, s_m[1:T])
    @variable(model, s_β[1:T])

    # especificando a dinâmica dos componentes atualizados
    #@constraint(model, )

    #@variable(model, -5.0 ≤ κ_m ≤ 5.0)
    #@variable(model, -5.0 ≤ κ_β ≤ 5.0)

    #passo de previsão
    m_pred = Vector(undef, T)
    β_pred = Vector(undef, T)
    for t in 2:T
        m_pred[t] = m_update[t-1] + β_update[t-1] +  s_m[t-1]
        β_pred[t] = β_update[t-1] + s_β[t-1]
    end

    @expression(model, μ, m_update)

    @constraint(model,[t = 2:T], s_m[t] == m_update[t] - m_pred[t])
    @constraint(model,[t = 2:T], s_β[t] == β_update[t] - β_pred[t])
    #@constraint(model,[t = 2:T], m_update[t] == m_update[t-1] + κ * score[t-1])

    f(y,μ, σ) = logpdf(Normal(μ,σ),y)
    @operator(model,ℓ,3,f)
    @objective(model, Max, sum(ℓ.(y,μ, σ)) - λ*s_β'I(T)*s_β - λ*s_m'I(T)*s_m)
    
    set_start_value.(m_update, y)
    set_start_value(σ, std(diff(y)))
    set_start_value(κ_m, 0.02)
    set_start_value(κ_β, 0.02)

    optimize!(model)

    println(termination_status(model))
    return value.(m_update), value.(m_pred[2:end]), value(σ), value.(s), value(ϕ), value(ω)
    
end

# Parâmetros
T = 100 
N = 10  
σ²_μ = 0.1   
σ²_ε = 1.0  

# Simulação
series = simulate_local_level(T, N, σ²_μ, σ²_ε)

y = series[:, 1]
plot(y)

λ = 0.5
μ_up, μ_pred, σ, s, κ = implicit_local_level_gas(y, λ);

plot(y[2:end], label = "série", color = "black")
plot!(μ_pred, label = "média preditiva", color = "red")
plot!(μ_up[2:end], label = "média atualizada", color = "blue") #atualizado mais suave do que o predito, o que faz sentido

#Testando a relação do score encontrada com condição de primeira ordem 

score_est = (y .- μ_up) ./ σ^2 #score da normal calculado com a média atualizada
score_alt = 2* λ .* (μ_up[2:end] .- μ_pred) 

plot(score_est[2:end])
plot!(score_alt)

#Não bate 

#Comparação com o pacote...
dist = UnobservedComponentsGAS.NormalDistribution()

model = UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, "random walk slope", "", missing)

rmse_model = zeros(N)
rmse_implicit_model = zeros(N)

time_model = zeros(N)
time_implicit_model = zeros(N)

for i in 1:N
    println("serie $i")
    y = series[:, i]

    plot(y)

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

    #CSV.write("results_figures/eta = $η/results.csv", DataFrame(results, ["serie", "time ESD", "time ISD", "rmse ESD", "rmse ISD"]))

hcat(time_model, time_implicit_model)
hcat(rmse_model, rmse_implicit_model)

value(ω)

plot(y[3:end])
plot!(value.(μ)[3:end])
plot!(value.(m_pred[3:end]))