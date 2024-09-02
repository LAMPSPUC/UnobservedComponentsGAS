using Distributions, JuMP, Ipopt, Plots, Random

# Modelo de nível local
import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/UnobservedComponentsGAS.jl")
using Distributions, JuMP, Ipopt, Plots, Random, CSV, DataFrames, LinearAlgebra

function simulate_local_level(T, N, σ²_μ, σ²_ε)

    # Inicialização das séries
    T = T + 5
    y = zeros(T, N) .+ rand(Uniform(0,100))
    
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
    
    return y[6:end,:]
end

function implicit_local_level_gas(y::Vector{Float64}, λ::Float64, d::Float64, η::Float64)
    T = length(y)

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 100000)
    set_optimizer_attribute(model, "max_cpu_time", 180.0)
    set_optimizer_attribute(model, "tol",  0.0005) 

    @variable(model, m_update[1:T])
    @variable(model, σ  ≥ 0.0004)
    @variable(model, m_pred1)
    @variable(model, -2 ≤ κ ≤ 2)
    @variable(model, s[1:T])

    #passo de previsão
    m_pred    = Vector(undef, T)
    m_pred[1] = m_pred1
    #s         = Vector(undef, T)
    #s[1]      = m_update[1] - m_pred[1]
    for t in 2:T
        m_pred[t] = m_update[t-1]
        #s[t]      = m_update[t] - m_pred[t]
    end

    #@constraint(model, [t in 1:T], s[t] == m_update[t] - m_pred[t])
    P = ones(T) * (1 / (λ*κ)) * (1/σ^2)^(d)

    f(y,μ, σ) = logpdf(Normal(μ,σ),y)
    @operator(model,ℓ,3,f)
    @objective(model, Max, sum(ℓ.(y,m_update, σ)) - (λ/2)*s'diagm(P)*s - η * sum((s[t] - m_update[t] + m_pred[t])^2 for t in 1:T))
    
    set_start_value.(m_update, y)
    #set_start_value(m_pred1, y[1])
    set_start_value(σ, std(diff(y)))
    set_start_value(κ, 0.02)

    optimize!(model)
    println(termination_status(model))
    return value.(m_update), value.(m_pred), value(σ), value.(s), value(κ), value.(P)
end

# Parâmetros
T = 100 
N = 10
σ²_μ = 10.0   
σ²_ε = 10.0  

# Simulação
series = simulate_local_level(T, N, σ²_μ, σ²_ε)

y = series[:, 6]
plot(y)
d_values = [0.0, 0.5, 1.0]

dict_results = Dict()
for i in 1:N
    println("serie $i")
    y = series[:, i]
    graficos_fit = Vector(undef, 3)
    graficos_score = Vector(undef, 3)
    for j in 1:3
        try
            d = d_values[j]
            println("d = $d")
            λ = 0.001
            dict_results[i] = Dict()
            μ_up, μ_pred, σ, s, κ, P_opt = implicit_local_level_gas(y, λ,d, 0.5);
           
            dist = UnobservedComponentsGAS.NormalDistribution()
            model = UnobservedComponentsGAS.GASModel(dist, [true, false], d, "random walk", "", missing)
            fitted_model = UnobservedComponentsGAS.fit(model, y)

            if d == 0
                score_implicit = s .* (λ * P_opt)[1]
                score_pack = (y .- fitted_model.fitted_params["param_1"]) ./ fitted_model.fitted_params["param_2"][1]
                score_up = (y .- μ_up) ./ σ^2 #score da normal calculado com a média atualizada
    
            elseif d == 0.5
                score_implicit = s .* (λ * P_opt * σ)[1]
                score_pack = (y .- fitted_model.fitted_params["param_1"]) ./ sqrt(fitted_model.fitted_params["param_2"][1])
                score_up = (y .- μ_up) ./ σ #score da normal calculado com a média atualizada
    
            else
                score_implicit  = s.* (λ * P_opt * σ^2)[1]
                score_pack = (y .- fitted_model.fitted_params["param_1"])
                score_up = (y .- μ_up)
    
            end

            dict_results[i]["μ_pred"] = μ_pred
            dict_results[i]["μ_up"] = μ_up
            dict_results[i]["σ2"] = σ^2
            dict_results[i]["score_implicit"] = score_implicit[2:end]
            dict_results[i]["κ"] = κ
            dict_results[i]["score_up"] = score_up[2:end]
            #dict_results[i]["score_alt"] = score_alt[2:end]
            dict_results[i]["score_pack"] = score_pack[2:end]
            dict_results[i]["σ2_pack"] = fitted_model.fitted_params["param_2"][1]
            dict_results[i]["μ_pack"] = fitted_model.fitted_params["param_1"]

            graficos_fit[j] = plot(y[3:end], label = "série", color = "black", linewidth = 1.5)
            plot!(dict_results[i]["μ_pred"][3:end], label = "média preditiva", color = "red", linewidth = 1.0)
            plot!(dict_results[i]["μ_up"][3:end], label = "média atualizada", color = "blue", linewidth = 1.0) #atualizado mais suave do que o predito, o que faz sentido
            plot!(dict_results[i]["μ_pack"][3:end], label = "média pacote", color = "green", linewidth = 1.0, legend =:bottomleft,  legendfontsize=5)
            title!("d = $d")
            #savefig("results_implicit_gas/medias_serie$i.png")

            graficos_score[j] = plot(dict_results[i]["score_implicit"], label = "score_implicit")
            plot!(dict_results[i]["score_up"], label = "score_up")
            #plot!(dict_results[λ]["s"], label = "score_opt")
            plot!(dict_results[i]["score_pack"], label = "score_pack", legend =:bottomleft, legendfontsize=8)
            title!("d = $d")
            #savefig("results_implicit_gas/scores_serie$i.png")
        catch
            println("Erro na serie $i")
        end
    end
    plot(graficos_fit[1], graficos_fit[2], graficos_fit[3], layout=(2, 2), suptitle = "Série $i")
    savefig("results_implicit_gas/medias_serie$i.png")

    plot(graficos_score[1], graficos_score[2], graficos_score[3], layout=(2, 2), suptitle = "Série $i")
    savefig("results_implicit_gas/scores_serie$i.png")
end





# O nosso pacote explode muito a variância
for λ in 0:0.1:1
    println("κ = ", dict_results[λ]["κ"])
    println("σ2 = ", dict_results[λ]["σ2"])
    println("σ2 pack = ", dict_results[λ]["σ2_pack"])
end


# A média do pacote é parecida com a média preditiva 
λ = 0.9
hcat(dict_results[λ]["μ_pred"][2:end],dict_results[λ]["μ_up"],dict_results[λ]["μ_pack"][2:end])

# Media atualizada está batendo com a média atualizada via conta.
# Meio óbvio por causa da restrição do modelo
plot(hcat(μ_up[2:end], μ_pred .+ (1/2*λ).*((y .- μ_up) ./ σ^2)[2:end])) # 

λ = 0.9
plot(y[2:end], label = "série", color = "black")
plot!(dict_results[λ]["μ_pred"], label = "média preditiva", color = "red")
plot!(dict_results[λ]["μ_up"][2:end], label = "média atualizada", color = "blue") #atualizado mais suave do que o predito, o que faz sentido

#Testando a relação do score encontrada com condição de primeira ordem 
#Não bate, mas tá seguindo o mesmo comportamento
plot(dict_results[0.6]["score_est"][2:end])
plot!(dict_results[0.6]["score_alt"])


#Comparação com o pacote...
dist = UnobservedComponentsGAS.NormalDistribution()

model = UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, "random walk", "", missing)

rmse_model = zeros(N)
rmse_implicit_model = zeros(N)

time_model = zeros(N)
time_implicit_model = zeros(N)

for i in 1:N
    println("serie $i")
    y = series[:, i]

    plot(y)

    time_model[i]          = @elapsed fitted_model = UnobservedComponentsGAS.fit(model, y)
    time_implicit_model[i] = @elapsed μ_up, μ_pred, σ, s, ϕ, ω = implicit_local_level_gas(y, 0.5)

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



# outras funções
# function simulate_local_linear_trend(T, N, σ²_μ, σ²_β, σ²_ε)
#     # Inicialização das séries
#     y = zeros(T, N)
    
#     # Inicialização dos estados
#     μ = zeros(T, N)
#     β = zeros(T, N)
    
#     # Distribuições dos ruídos
#     d_ημ = Normal(0, sqrt(σ²_μ))
#     d_ηβ = Normal(0, sqrt(σ²_β))
#     d_ε = Normal(0, sqrt(σ²_ε))
    
#     # Condições iniciais
#     μ[1, :] = randn(N)
#     β[1, :] = randn(N)
    
#     for n in 1:N
#         for t in 2:T
#             # Atualização dos estados
#             μ[t, n] = μ[t-1, n] + β[t-1, n] + rand(d_ημ)
#            β[t, n] = β[t-1, n] + rand(d_ηβ)
            
#             # Geração da observação
#             y[t, n] = μ[t, n] + rand(d_ε)
#         end
#     end
    
#     return y
# end

# function implicit_local_level_gas(y::Vector{Float64}, λ::Float64)
#     T = length(y)

#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     set_optimizer_attribute(model, "max_iter", 60000)
#     set_optimizer_attribute(model, "max_cpu_time", 120.0)
#     set_optimizer_attribute(model, "tol",  0.0005) 

#     @variable(model, m_update[1:T])
#     @variable(model, σ  ≥ 0.0004)
#     @variable(model, m_pred1)
#     @variable(model, s[1:T])
#     @variable(model, -2 ≤ κ ≤ 2)

#     #passo de previsão
#     m_pred    = Vector(undef, T)
#     m_pred[1] = m_pred1
#     #s         = Vector(undef, T)
#     #s[1]      = m_update[1] - m_pred[1]
#     for t in 2:T
#         m_pred[t] = m_update[t-1]
#         #s[t]      = m_update[t] - m_pred[t]
#     end

#     @expression(model, μ, m_update)
#     @constraint(model, [t = 1:T], m_update[t] == m_pred[t] + κ*s[t])
    
#     f(y,μ, σ) = logpdf(Normal(μ,σ),y)
#     @operator(model,ℓ,3,f)
#     @objective(model, Max, sum(ℓ(y[t],μ[t], σ) for t = 1:T) - (λ/2)*sum(s[t]^2 for t = 1:T))
    
#     set_start_value.(m_update, y)
#     #set_start_value(m_pred1, y[1])
#     set_start_value(σ, std(diff(y)))
#     set_start_value(κ, 0.02)

#     optimize!(model)
#     println(termination_status(model))

#     return value.(m_update), value.(m_pred), value(σ), value.(s), value(κ)
    
# end

# function implicit_local_linear_trend_gas(y::Vector{Float64}, λ::Float64)
#     T = length(y)

#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     set_optimizer_attribute(model, "max_iter", 30000)
#     set_optimizer_attribute(model, "max_cpu_time", 100.0)
#     set_optimizer_attribute(model, "tol",  0.005) 

#     @variable(model, m_update[1:T])
#     @variable(model, β_update[1:T])
#     @variable(model, σ  ≥ 0.0004)
#     @variable(model, s_m[1:T])
#     @variable(model, s_β[1:T])

#     # especificando a dinâmica dos componentes atualizados
#     #@constraint(model, )

#     #@variable(model, -5.0 ≤ κ_m ≤ 5.0)
#     #@variable(model, -5.0 ≤ κ_β ≤ 5.0)

#     #passo de previsão
#     m_pred = Vector(undef, T)
#     β_pred = Vector(undef, T)
#     for t in 2:T
#         m_pred[t] = m_update[t-1] + β_update[t-1] +  s_m[t-1]
#         β_pred[t] = β_update[t-1] + s_β[t-1]
#     end

#     @expression(model, μ, m_update)

#     @constraint(model,[t = 2:T], s_m[t] == m_update[t] - m_pred[t])
#     @constraint(model,[t = 2:T], s_β[t] == β_update[t] - β_pred[t])
#     #@constraint(model,[t = 2:T], m_update[t] == m_update[t-1] + κ * score[t-1])

#     f(y,μ, σ) = logpdf(Normal(μ,σ),y)
#     @operator(model,ℓ,3,f)
#     @objective(model, Max, sum(ℓ.(y,μ, σ)) - λ*s_β'I(T)*s_β - λ*s_m'I(T)*s_m)
    
#     set_start_value.(m_update, y)
#     set_start_value(σ, std(diff(y)))
#     set_start_value(κ_m, 0.02)
#     set_start_value(κ_β, 0.02)

#     optimize!(model)

#     println(termination_status(model))
#     return value.(m_update), value.(m_pred[2:end]), value(σ), value.(s), value(ϕ), value(ω)
    
# end