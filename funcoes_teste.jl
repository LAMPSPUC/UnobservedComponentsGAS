module FuncoesTeste

using CSV, DataFrames
using Distributions, SpecialFunctions
using JuMP, Ipopt


function fit_harmonics(y::Vector{Fl}, seasonal_period::Int64, stochastic::Bool) where {Fl}

    T = length(y)

    if seasonal_period % 2 == 0
        num_harmonic = Int64(seasonal_period / 2)
    else
        num_harmonic = Int64((seasonal_period -1) / 2)
    end

    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    set_optimizer_attribute(model, "print_level", 0)
    #set_optimizer_attribute(model, "hessian_constant", "yes")

    @variable(model, y_hat[1:T])

    if stochastic
        @variable(model, γ[1:num_harmonic, 1:T])
        @variable(model, γ_star[1:num_harmonic, 1:T])

        @constraint(model, [i = 1:num_harmonic, t = 2:T], γ[i, t] == γ[i, t-1] * cos(2*π*i / seasonal_period) + 
                                                                    γ_star[i,t-1]*sin(2*π*i / seasonal_period))
        @constraint(model, [i = 1:num_harmonic, t = 2:T], γ_star[i, t] == -γ[i, t-1] * sin(2*π*i / seasonal_period) + 
                                                                                γ_star[i,t-1]*cos(2*π*i / seasonal_period))

        @constraint(model, [t = 1:T], y_hat[t] == sum(γ[i, t] for i in 1:num_harmonic))
    else

        @variable(model, γ[1:num_harmonic])
        @variable(model, γ_star[1:num_harmonic])

        @constraint(model, [t = 1:T], y_hat[t] == sum(γ[i] * cos(2 * π * i * t/seasonal_period) + 
                                                  γ_star[i] * sin(2 * π * i* t/seasonal_period)  for i in 1:num_harmonic))
    end
   
    @objective(model, Min, sum((y .- y_hat).^2))
    optimize!(model)

    return JuMP.value.(γ), JuMP.value.(γ_star)
    
end


function get_initial_α(y::Vector{Float64})

    T = length(y)
    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, α >= 1e-2)  # Ensure α is positive
    @variable(model, λ[1:T] .>= 1e-4)
    register(model, :Γ, 1, Γ; autodiff = true)
    @NLobjective(model, Max, sum(-log(Γ(α)) - α*log(1/α) - α*log(λ[i]) +(α-1)*log(y[i]) - (α/λ[i])*y[i] for i in 1:T))
    optimize!(model)
    if has_values(model)
        return JuMP.value.(α)
    else
        return fit_mle(Gamma, y).α
    end 
end


function get_initial_params_lognormal(y::Vector{Fl}, time_varying_params::Vector{Bool}) where Fl

    y = log.(y)
    T         = length(y)
    # dist_code = get_dist_code(dist)
    seasonal_period = 12#get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist) #(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = var(diff(y))#get_initial_σ(y)^2#
    end

    return initial_params
end


"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"
function get_initial_params_gamma(y::Vector{Fl}, time_varying_params::Vector{Bool}) where Fl

    println("Inicialização dos parâmetros iniciais")
    T         = length(y)
    # dist_code = get_dist_code(dist)
    seasonal_period = 12#get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()
    fitted_distribution = fit_mle(Gamma, y)
    
    # param[2] = λ = média
    if time_varying_params[2]
        println("λ = y")
        initial_params[2] = y
    else
        println("λ = mean(y)")
        initial_params[2] = fitted_distribution.α*fitted_distribution.θ
    end

    # param[1] = α
    if time_varying_params[1]
        println("α = ??")
        initial_params[1] = get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        println("α = $(fitted_distribution.α)")
        println(("α = $(get_initial_α(y))"))
        initial_params[1] = get_initial_α(y)#mean(y)^2/var((y)) 
    end
    
    return initial_params
end
 
 

function get_initial_values_from_components(y, components, stochastic, dist)
    
    T = size(components, 1)
    initial_rws         = components[:,"l"]
    initial_slope       = components[:,"b"]
    initial_seasonality = components[:,"s1"]
    initial_ar          = zeros(T)
    
    initial_values                          = Dict{String,Any}()
    if dist == "Gamma" 
        initial_params = get_initial_params_gamma(y, [false, true])
        initial_values["fixed_param"]           = [initial_params[1]]
        initial_values["param"]                 = initial_params[2]
    else
        initial_params = get_initial_params_lognormal(y, [true, false])
        initial_values["fixed_param"]           = [initial_params[2]]
        initial_values["param"]                 = initial_params[1]
    end

    initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, 12, stochastic)

   

    

    initial_values["intercept"]             = Dict()
    initial_values["intercept"]["values"]   = [0.02]
    initial_values["rw"]                    = Dict()
    initial_values["rw"]["values"]          = zeros(T)
    initial_values["rw"]["κ"]               = 0.02
    initial_values["rws"]                   = Dict()
    initial_values["rws"]["values"]         = initial_rws
    initial_values["rws"]["κ"]              = 0.02
    initial_values["slope"]                 = Dict()
    initial_values["slope"]["values"]       = initial_slope
    initial_values["slope"]["κ"]            = 0.02
    initial_values["seasonality"]           = Dict()
    initial_values["seasonality"]["values"] = initial_seasonality
    initial_values["seasonality"]["γ"]      = initial_γ
    initial_values["seasonality"]["γ_star"] = initial_γ_star
    initial_values["seasonality"]["κ"]      = 0.02
    initial_values["ar"]                    = Dict()
    initial_values["ar"]["values"]          = initial_ar
    initial_values["ar"]["ϕ"]               = [0.]
    initial_values["ar"]["κ"]               = 0.02
    return initial_values
end


end