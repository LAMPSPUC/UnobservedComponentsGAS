using Random, CSV, DataFrames, JuMP, Ipopt, Distributions, Plots

function generate_local_level_series(N, T, steps_ahead)

    series_train = zeros(N,T)
    series_test  = zeros(N, steps_ahead)
    σ² = 2.0
    κ  = 1.0
    Random.seed!(0)
    for i in 1:N
        μ0 = 5.0
        s = zeros(T + steps_ahead+1)
        y = zeros(T + steps_ahead)
        μ = vcat(μ0,zeros(T + steps_ahead))

        for t in 1:T + steps_ahead
            y[t] = rand(Normal(μ[t], sqrt(σ²)))
            s[t] = (y[t] - μ[t]) / σ²
            μ[t+1] = μ[t] + κ * s[t]
        end

        #plot(y)
        series_train[i, :] = y[1:T]
        series_test[i, :] = y[T+1:end]
    end

    return series_train, series_test
end

function model_with_s_regularization(y, d)

    T = length(y)
    model = JuMP.Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 100.0)
    set_optimizer_attribute(model, "tol",  0.005) 
    set_optimizer_attribute(model, "print_level", 5)  # Adiciona mais detalhes no output

    @variable(model, μ[1:T])
    @variable(model, σ² ≥ 0.0001)
    @variable(model, -2 ≤ κ ≤ 2)
    @variable(model, s[1:T])

    @constraint(model, [t = 2:T], μ[t] == μ[t-1] + κ * s[t-1], base_name = "con")
    #@constraint(model, [t = 1:T], s[t] == (y[t] - μ[t]) ./ σ²)
    if d == 1.0
        @expression(model, λ, κ / (2 * σ²))
    elseif d == 0.5
        @expression(model, λ, κ / (2 * sqrt(σ²)))
    else
        @expression(model, λ, κ / 2)
    end

    #@expression(model, γ, sum((y[t] - μ[t]) * s[t]  for t = 1:T) / (2* σ² * κ))

    @objective(model, Min, -sum(-0.5 * log(2 * π * σ²) - 0.5 * ((y[t] - μ[t])^2 / σ²) for t in 1:T) + sum(λ * s[t]^2 for t = 1:T))

    set_start_value.(μ, y)
    set_start_value(σ²,  var(diff(y)))
    set_start_value(κ, 0.02)
    optimize!(model)

    α_values = zeros(T)
    for t in 2:T
        α_values[t] = dual(constraint_by_name(model, "con[$t]"))
    end
    println(termination_status(model))

    return value.(μ), value(σ²), value.(s), value(κ), α_values
end

function compute_score(μ, σ2, y, d)
    if d == 1.0
        return (y - μ)
    elseif d == 0.5
        return (y - μ) / sqrt(σ2)
    else
        return (y - μ) / σ2
    end
end

# function model_with_s_and_κ_regularizations(y, d)

#     T = length(y)
#     model = JuMP.Model(Ipopt.Optimizer)

#     set_optimizer_attribute(model, "max_iter", 30000)
#     set_optimizer_attribute(model, "max_cpu_time", 300.0)
#     set_optimizer_attribute(model, "tol",  0.005)

#     @variable(model, μ[1:T])
#     @variable(model, σ² ≥ 0.0001)
#     @variable(model, -2.0 ≤ κ ≤ 2.0)
#     @variable(model, s[1:T])

#     @constraint(model, [t = 2:T], μ[t] == μ[t-1] + κ * s[t-1])
#     if d == 1.0
#         @expression(model, λ, κ / (2 * σ²))
#     elseif d == 0.5
#         @expression(model, λ, κ / (2 * √σ²))
#     else
#         @expression(model, λ, κ / 2)
#     end

#     @expression(model, γ, sum((y[t] - μ[t]) * s[t]  for t = 1:T) / (2* σ² * κ))

#     @objective(model, Min, -sum(-0.5 * log(2 * π * σ²) - 0.5 * ((y[t] - μ[t])^2 / σ²) for t in 1:T) + sum(λ * s[t]^2 for t = 1:T) + γ * κ^2)

#     set_start_value.(μ, y)
#     set_start_value(σ²,  var(diff(y)))
#     set_start_value(κ, 0.02)
#     optimize!(model)

#     println(termination_status(model))

#     return value.(μ), value(σ²), value.(s), value(κ)
# end

# function model_with_sκ_regularization(y, d)

#     T = length(y)
#     model = JuMP.Model(Ipopt.Optimizer)

#     set_optimizer_attribute(model, "max_iter", 30000)
#     set_optimizer_attribute(model, "max_cpu_time", 300.0)
#     set_optimizer_attribute(model, "tol",  0.005)

#     @variable(model, μ[1:T])
#     @variable(model, σ² ≥ 0.0001)
#     @variable(model, -2.0 ≤ κ ≤ 2.0)
#     @variable(model, s[1:T])

#     @constraint(model, [t = 2:T], μ[t] == μ[t-1] + κ * s[t-1])
#     if d == 1.0
#         @expression(model, λ, 1 / (2 * κ * σ²))
#     elseif d == 0.5
#         @expression(model, λ, 1 / (2 * κ * √σ²))
#     else
#         @expression(model, λ, 1 / (2 * κ))
#     end

#     #@expression(model, γ, sum((y[t] - μ[t]) * s[t]  for t = 1:T) / (2* σ² * κ))

#     @objective(model, Min, -sum(-0.5 * log(2 * π * σ²) - 0.5 * ((y[t] - μ[t])^2 / σ²) for t in 1:T) + κ^2 * sum(λ * s[t]^2 for t = 1:T))

#     set_start_value.(μ, y)
#     set_start_value(σ²,  var(diff(y)))
#     set_start_value(κ, 0.02)
#     optimize!(model)

#     println(termination_status(model))

#     return value.(μ), value(σ²), value.(s), value(κ)
# end

train, test = generate_local_level_series(100, 144, 24)
y = train[1, :]
plot(y)

d = 0.0
μ1, σ1, s1, κ1, α1 = model_with_s_regularization(y,d)

score_est1 = compute_score.(μ1, σ1, y, 1.0)
score_est05 = compute_score.(μ1, σ1, y, 0.5)
score_est0 = compute_score.(μ1, σ1, y, 0.0)

hcat(s1, score_est1)


sum(α1 .* s1)

plot(y)
plot!(μ1, label = "model 1")
plot!(μ2, label = "model 2")
plot!(μ3, label = "model 3")

s = (y .- μ1) ./ σ1

hcat(α1, s)


hcat(s1, s)

