get_idxs_time_varying_params(time_varying_params::Vector{Bool}) = findall(i -> i == true, time_varying_params)

"
Returns a dictionary with the fitted hyperparameters and components.
"
function get_fitted_values(gas_model::GASModel, model::Ml, X::Union{Missing, Matrix{Fl}}) where {Ml,  Fl} 
    
    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    idx_params = get_idxs_time_varying_params(time_varying_params)

    # Getting Fit in sample
    if 1 ∈ idx_params
        fit_in_sample = Vector(value.(model[:params][:, 1]))
    else
        fit_in_sample = ones(size(model[:params], 1)) * value(model[:fixed_params][1])
    end


    # Getting fitted parameters
    fitted_params = Dict{String, Vector{Float64}}()
    for i in eachindex(time_varying_params)
        if time_varying_params[i]
            fitted_params["param_$i"] = value.(model[:params][:, i])
        else
            fitted_params["param_$i"] = ones(length(fit_in_sample)) * value(model[:fixed_params][i])
        end
    end
    
    components = Dict{String, Any}()

    for i in idx_params

        components["param_$i"] = Dict{String, Any}()
        components["param_$i"]["intercept"] = value(model[:c][i])

        if has_random_walk(random_walk, i)
            components["param_$i"]["level"]                    = Dict{String, Any}()
            components["param_$i"]["level"]["hyperparameters"] = Dict{String, Any}()

            components["param_$i"]["level"]["value"]                = Vector(value.(model[:RW][:, i]))
            components["param_$i"]["level"]["hyperparameters"]["κ"] = value(model[:κ_RW][i])
        end

        if has_random_walk_slope(random_walk_slope, i)
            components["param_$i"]["level"]                    = Dict{String, Any}()
            components["param_$i"]["level"]["hyperparameters"] = Dict{String, Any}()
            components["param_$i"]["slope"]                    = Dict{String, Any}()
            components["param_$i"]["slope"]["hyperparameters"] = Dict{String, Any}()

            components["param_$i"]["level"]["value"]                = Vector(value.(model[:RWS][:, i]))
            components["param_$i"]["level"]["hyperparameters"]["κ"] = value(model[:κ_RWS][i])
            components["param_$i"]["slope"]["value"]                = Vector(value.(model[:b][:, i]))
            components["param_$i"]["slope"]["hyperparameters"]["κ"] = value(model[:κ_b][i])
        end

        if has_seasonality(seasonality, i)
            components["param_$i"]["seasonality"]                    = Dict{String, Any}()
            components["param_$i"]["seasonality"]["hyperparameters"] = Dict{String, Any}()

            components["param_$i"]["seasonality"]["value"]                     = Vector(value.(model[:S][:, i]).data)
            if stochastic
                components["param_$i"]["seasonality"]["hyperparameters"]["γ"]      = Matrix(value.(model[:γ][:,:, i]))
                components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"] = Matrix(value.(model[:γ_star][:,:, i]))
                components["param_$i"]["seasonality"]["hyperparameters"]["κ"]      = value(model[:κ_S][i])
            else
                components["param_$i"]["seasonality"]["hyperparameters"]["γ"]      = Vector(value.(model[:γ][:, i]))
                components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"] = Vector(value.(model[:γ_star][:, i]))
            end

        end

        if has_AR(ar, i)
            components["param_$i"]["ar"]                    = Dict{String, Any}()
            components["param_$i"]["ar"]["hyperparameters"] = Dict{String, Any}()

            components["param_$i"]["ar"]["value"]                = Vector(value.(model[:AR][:, i]))
            components["param_$i"]["ar"]["hyperparameters"]["κ"] = value(model[:κ_AR][i])
            components["param_$i"]["ar"]["hyperparameters"]["ϕ"] = Vector(value.(model[:ϕ][:, i]))
        end

        if !ismissing(X) && i == 1
            components["param_$i"]["explanatories"] = Vector(value.(model[:β][:, i]))
        end

    end

    return fit_in_sample, fitted_params, components

end

"Compute the Standard Residuals of the fitted model"
function get_std_residuals(y::Vector{Fl}, fit_in_sample::Vector{Fl}) where Fl
    residuals = y .- fit_in_sample
    std_res = (residuals .- mean(residuals)) / std(residuals)

    return std_res
end

"Compute the Conditional Score Residuals of the fitted model"
function get_cs_residuals(y::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64) where Fl
    
    T = length(y)
    num_params = length(fitted_params)  ## COMO É ESSA CONTA?

    cs_residuals = zeros(T, num_params)

    if num_params == 2
        for i in 1:num_params
            for t in 1:T
                cs_residuals[t, i] = scaled_score(fitted_params["param_1"][t], fitted_params["param_2"][t], y[t], 0.5, dist_code, i)
            end
        end
    elseif num_params == 3
        for i in 1:(num_params-1)
            for t in 1:T
                cs_residuals[t, i] = scaled_score(fitted_params["param_1"][t], fitted_params["param_2"][t],fitted_params["param_3"][t], y[t], 0.5, dist_code, i)
            end
        end
    else
        #INCLUIR CASO COM 1 PARAMETRO
    end

    return cs_residuals
end

"Compute the Quantile Residuals of the fitted model"
function get_quantile_residuals(y::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64) where Fl
    
    dist_name  = DICT_CODE[dist_code]
    T          = length(y)
    num_params = length(fitted_params)

    q_residuals = zeros(T)
    if num_params == 2
        for t in 1:T
            PIT = DICT_CDF[dist_name]([fitted_params["param_1"][t],fitted_params["param_2"][t]],  y[t])
            q_residuals[t] = quantile(Normal(0, 1), PIT)
        end
    elseif num_params == 3
        for t in 1:T
            PIT = DICT_CDF[dist_name]([fitted_params["param_1"][t],fitted_params["param_2"][t], fitted_params["param_3"][t]],  y[t])
            q_residuals[t] = quantile(Normal(0, 1), PIT)
        end
    end

    return q_residuals
end

"
Returns the residuals of the fitted model.
"
function get_residuals(y::Vector{Float64}, fit_in_sample::Vector{Float64}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64)

    # Getting std residuals
    std_res = get_std_residuals(y, fit_in_sample)

    # Getting Conditional Score Residuals 
    cs_residuals = get_cs_residuals(y, fitted_params, dist_code)

    # Getting Quantile Residuals
    q_residuals = get_quantile_residuals(y, fitted_params, dist_code)

    dict_residuals = Dict{String, Union{Vector{Float64}, Matrix{Float64}}}()
    dict_residuals["std_residuals"] = std_res
    dict_residuals["cs_residuals"]  = cs_residuals
    dict_residuals["q_residuals"]   = q_residuals

    return dict_residuals
    
end

"
Calculates the AIC information criteria.
"
function aic(model::Ml,  parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

    #num_params = get_num_params(dist)
    dist_code = get_dist_code(dist)

    num_parameters = length(all_variables(model))

    dist_name = DICT_CODE[dist_code]

    log_likelihood = 0
    for i in 2:length(y)
        log_likelihood += DICT_LOGPDF[dist_name](value.(parameters[i, :]), y[i])
    end

    return -2 * log_likelihood + 2 * num_parameters
end

"
Calculates the BIC information criteria.
"
function bic(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}
    
    #num_params = get_num_params(dist)
    dist_code = get_dist_code(dist)

    num_parameters = length(all_variables(model))

    dist_name = DICT_CODE[dist_code]
    log_likelihood = 0
    for i in 2:length(y)
        log_likelihood += DICT_LOGPDF[dist_name](value.(parameters[i, :]), y[i])
    end

    return -2 * log_likelihood + num_parameters * log(length(y[2:end]))
end

"
Calculates the AICc information criteria.
"
function aicc(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

    #num_params = get_num_params(dist)
    
    num_parameters = length(all_variables(model))

    AIC = aic(model, parameters, y, dist)

    return AIC + (2 *  num_parameters * ( num_parameters + 1)) / (length(y[2:end]) -  num_parameters - 1)
end

"
Returns a dictionary with the information criteria AIC, BIC and AICc.
"
function get_information_criteria(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

    dict = Dict{String, Float64}()

    dict["aic"]  = aic(model, parameters, y, dist)
    dict["bic"]  = bic(model, parameters, y, dist)
    dict["aicc"] = aicc(model, parameters, y, dist)

    return dict
end

"
Check if a fitte model find a optimal or presented Invalid model or numeric error
"
function is_valid_model(output::Output)

    return output.model_status ∉ ["INVALID_MODEL", "NUMERIC_ERROR", "TIME_LIMIT"]
end

function fit_AR_model(y::Vector{Fl}, order::Vector{Int64}) where Fl

    T         = length(y)
    max_order = maximum(order) 

    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    set_optimizer_attribute(model, "print_level", 0)

    @variable(model, c)
    @variable(model, ϕ[order])
    @variable(model, y_hat[1:T])

    @constraint(model, [t = max_order+1:T], y_hat[t] == c + sum(ϕ[i]*y[t - i] for i in order))
   
    @objective(model, Min, sum((y .- y_hat).^2))
    optimize!(model)

    return JuMP.value.(y_hat), JuMP.value.(ϕ), JuMP.value(c)
end

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

    return value.(γ), value.(γ_star)
    
end

