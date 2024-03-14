"""
# get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, steps_ahead::Int64, num_scenarios::Int64)

Creates a dictionary containing hyperparameters and fitted components along with null forecasts for a GAS model.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `steps_ahead::Int64`: The number of steps ahead for the forecast.
- `num_scenarios::Int64`: The number of scenarios for the forecast.

## Returns
- `dict_hyperparams_and_fitted_components`: A dictionary containing hyperparameters and fitted components with null forecasts that will be filled in the function predict_scenarios.
"""
function get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, steps_ahead::Int64, num_scenarios::Int64)

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    idx_params = get_idxs_time_varying_params(time_varying_params) 
    order      = get_AR_order(ar)
    num_params = get_num_params(dist)
    components = output.components

    num_harmonic, seasonal_period = UnobservedComponentsGAS.get_num_harmonic_and_seasonal_period(seasonality)

    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    elseif length(idx_params) > length(num_harmonic) #considera os mesmos harmonicos para todos os parametros variantes, para não quebrar a update_S!
        num_harmonic = Int64.(ones(length(idx_params)) * num_harmonic[1])
    end
    
    T_fitted = length(output.fitted_params["param_1"])

    dict_hyperparams_and_fitted_components                = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["rw"]          = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["rws"]         = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["seasonality"] = Dict{String, Any}()    
    dict_hyperparams_and_fitted_components["ar"]          = Dict{String, Any}()

    dict_hyperparams_and_fitted_components["params"]       = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["intercept"]    = zeros(num_params)
    dict_hyperparams_and_fitted_components["score"]        = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
  
    dict_hyperparams_and_fitted_components["rw"]["value"]  = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rw"]["κ"]      = zeros(num_params)

    dict_hyperparams_and_fitted_components["rws"]["value"] = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rws"]["b"]     = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rws"]["κ"]     = zeros(num_params)
    dict_hyperparams_and_fitted_components["rws"]["κ_b"]   = zeros(num_params)

    dict_hyperparams_and_fitted_components["seasonality"]["value"]   = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["seasonality"]["κ"]       = zeros(num_params)

    if stochastic
        dict_hyperparams_and_fitted_components["seasonality"]["γ"]       = zeros(num_harmonic[idx_params[1]], T_fitted + steps_ahead, num_params, num_scenarios) 
        dict_hyperparams_and_fitted_components["seasonality"]["γ_star"]  = zeros(num_harmonic[idx_params[1]], T_fitted + steps_ahead, num_params, num_scenarios)
    else
        dict_hyperparams_and_fitted_components["seasonality"]["γ"]       = zeros(num_harmonic[idx_params[1]], num_params) 
        dict_hyperparams_and_fitted_components["seasonality"]["γ_star"]  = zeros(num_harmonic[idx_params[1]], num_params)
    end
    
    dict_hyperparams_and_fitted_components["ar"]["value"]  = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["ar"]["ϕ"]      = zeros(maximum(vcat(order...)), num_params)
    dict_hyperparams_and_fitted_components["ar"]["κ"]      = zeros(num_params)

    for i in 1:num_params
        dict_hyperparams_and_fitted_components["params"][i, 1:T_fitted, :] .= output.fitted_params["param_$i"]

        if i in idx_params
            dict_hyperparams_and_fitted_components["intercept"][i] = components["param_$i"]["intercept"]
        end

        if has_random_walk(random_walk, i)
            dict_hyperparams_and_fitted_components["rw"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["level"]["value"]
            dict_hyperparams_and_fitted_components["rw"]["κ"][i]                     = components["param_$i"]["level"]["hyperparameters"]["κ"]
        end

        if has_random_walk_slope(random_walk_slope, i)
            dict_hyperparams_and_fitted_components["rws"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["level"]["value"]
            dict_hyperparams_and_fitted_components["rws"]["κ"][i]                     = components["param_$i"]["level"]["hyperparameters"]["κ"]
            dict_hyperparams_and_fitted_components["rws"]["b"][i, 1:T_fitted, :]     .= components["param_$i"]["slope"]["value"]
            dict_hyperparams_and_fitted_components["rws"]["κ_b"][i]                   = components["param_$i"]["slope"]["hyperparameters"]["κ"]
        end

        if has_AR(ar, i)
            dict_hyperparams_and_fitted_components["ar"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["ar"]["value"]
            dict_hyperparams_and_fitted_components["ar"]["ϕ"][:, i]                  = components["param_$i"]["ar"]["hyperparameters"]["ϕ"]
            dict_hyperparams_and_fitted_components["ar"]["κ"][i]                     = components["param_$i"]["ar"]["hyperparameters"]["κ"]
        end 

        if has_seasonality(seasonality, i)
            dict_hyperparams_and_fitted_components["seasonality"]["value"][i, 1:T_fitted, :]    .= components["param_$i"]["seasonality"]["value"]
            # dict_hyperparams_and_fitted_components["seasonality"]["κ"][i]                        = components["param_$i"]["seasonality"]["hyperparameters"]["κ"]
            if stochastic
                dict_hyperparams_and_fitted_components["seasonality"]["κ"][i]                        = components["param_$i"]["seasonality"]["hyperparameters"]["κ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, 1:T_fitted, i, :]     .= components["param_$i"]["seasonality"]["hyperparameters"]["γ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][:, 1:T_fitted, i,:] .= components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"]
            else
                # println(dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, i])
                # println(components["param_$i"]["seasonality"]["hyperparameters"]["γ"])
                dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, i]       .= components["param_$i"]["seasonality"]["hyperparameters"]["γ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][:, i,] .= components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"]
            end

        end
    end

    if length(idx_params) != num_params
        for i in setdiff(1:num_params, idx_params)
            dict_hyperparams_and_fitted_components["params"][i, :, :] .= output.fitted_params["param_$i"][1] #### PQ O [1] NO FINAL?
        end
    end

    return dict_hyperparams_and_fitted_components
end

"""
# get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64)

Creates a dictionary that includes hyperparameters, fitted components, null forecasts, and incorporates the effects of explanatory variables for a GAS model.
## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `X_forecast::Matrix{Fl}`: The matrix of explanatory variables for forecasting.
- `steps_ahead::Int64`: The number of steps ahead for the forecast.
- `num_scenarios::Int64`: The number of scenarios for the forecast.

## Returns
- `dict_hyperparams_and_fitted_components`: A dictionary containing hyperparameters, fitted components, forecasts, and explanatory variables.
"""
function get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64) where {Fl}
    
    n_exp      = size(X_forecast, 2)
    num_params = get_num_params(gas_model.dist)
    components = output.components

    dict_hyperparams_and_fitted_components = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, output, steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["explanatories"] = zeros(n_exp, num_params)

    dict_hyperparams_and_fitted_components["explanatories"][:, 1] = components["param_1"]["explanatories"]
    
    return dict_hyperparams_and_fitted_components

end

"""
# update_score!(dict_hyperparams_and_fitted_components::Dict{String, Any}, pred_y::Matrix{Float64}, d::Float64, dist_code::Int64, param::Int64, t::Int64, s::Int64)

Updates the score predictions for the specified parameter, period of time and scenario in the dict_hyperparams_and_fitted_components object.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `pred_y::Matrix{Float64}`: The matrix of predicted values for the time series data.
- `d::Float64`: The degree of freedom parameter.
- `dist_code::Int64`: The code representing the distribution used in the GAS model.
- `param::Int64`: The index of the parameter for which the score is being updated.
- `t::Int64`: The time step at which the score is being updated.
- `s::Int64`: The scenario index.

Return
Updates the score predictions for the specified parameter, period of time and scenario in the dict_hyperparams_and_fitted_components object.
"""
function update_score!(dict_hyperparams_and_fitted_components::Dict{String, Any}, pred_y::Matrix{Float64}, d::Float64, dist_code::Int64, param::Int64, t::Int64, s::Int64)

    if size(dict_hyperparams_and_fitted_components["params"])[1] == 2
        dict_hyperparams_and_fitted_components["score"][param, t, s] = scaled_score(dict_hyperparams_and_fitted_components["params"][1, t - 1, s], 
                                                                                dict_hyperparams_and_fitted_components["params"][2, t - 1, s], 
                                                                                pred_y[t - 1, s], d, dist_code, param)
    elseif size(dict_hyperparams_and_fitted_components["params"])[1] == 3
        dict_hyperparams_and_fitted_components["score"][param, t, s] = scaled_score(dict_hyperparams_and_fitted_components["params"][1, t - 1, s], 
                                                                                dict_hyperparams_and_fitted_components["params"][2, t - 1, s],
                                                                                dict_hyperparams_and_fitted_components["params"][3, t - 1, s], 
                                                                                pred_y[t - 1, s], d, dist_code, param)
    end
end

"""
# update_rw!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

Updates the random walk component of a dictionary containing hyperparameters and fitted components.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `param::Int64`: The index of the parameter for which the score is being updated.
- `t::Int64`: The time step at which the score is being updated.
- `s::Int64`: The scenario index.

## Returns
Updates the predictions of the random walk component for the specified parameter, time period, and scenario within the dict_hyperparams_and_fitted_components object.
"""
function update_rw!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

    dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] = dict_hyperparams_and_fitted_components["rw"]["value"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rw"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s] 
end

"""
# update_rws!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

Updates the random walk component of a dictionary containing hyperparameters and fitted components.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `param::Int64`: The index of the parameter for which the score is being updated.
- `t::Int64`: The time step at which the score is being updated.
- `s::Int64`: The scenario index.

## Returns
Updates the predictions of the random walk  plus slope component for the specified parameter, time period, and scenario within the dict_hyperparams_and_fitted_components object.
"""
function update_rws!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)
    
    dict_hyperparams_and_fitted_components["rws"]["b"][param, t, s] = dict_hyperparams_and_fitted_components["rws"]["b"][param, t - 1, s] + 
                                                                                    dict_hyperparams_and_fitted_components["rws"]["κ_b"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]

    dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] = dict_hyperparams_and_fitted_components["rws"]["value"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rws"]["b"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rws"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]

end

"""
# update_S!(dict_hyperparams_and_fitted_components::Dict{String, Any}, num_harmonic::Vector{Int64}, diff_T::Int64, param::Int64, t::Int64, s::Int64)

Updates the seasonality component predictions for the specified parameter, time period, and scenario in the `dict_hyperparams_and_fitted_components` object.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `num_harmonic::Vector{Int64}`: A vector specifying the number of harmonics for each parameter.
- `diff_T::Int64`: An integer representing the difference between the current time period and the base time period for seasonality calculations.
- `param::Int64`: An integer indicating the index of the parameter for which the seasonality component is being updated.
- `t::Int64`: An integer indicating the time step at which the seasonality component is being updated.
- `s::Int64`: An integer indicating the scenario index.

## Returns
Updates the seasonality component predictions for the specified parameter, time period, and scenario in the `dict_hyperparams_and_fitted_components` object.
"""
# Testar se a função precisa mesmo do diff_T  na sazo deterministica. Eu acho que não pq ja passamos o t = T + t
function update_S!(dict_hyperparams_and_fitted_components::Dict{String, Any}, num_harmonic::Vector{Int64}, diff_T::Int64, param::Int64, t::Int64, s::Int64)

    if length(size( dict_hyperparams_and_fitted_components["seasonality"]["γ"])) == 4
        for j in 1:num_harmonic[param]
            dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t, param, s] = dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t-1, param, s]*cos(2 * π * j /(num_harmonic[param] * 2)) +
                                                                                            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, t-1, param, 1]*sin(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                            dict_hyperparams_and_fitted_components["seasonality"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]

            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, t, param, s] = -dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t-1,param, s]*sin(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j,t-1, param, s]*cos(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                                dict_hyperparams_and_fitted_components["seasonality"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
        end

        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t, param, s] for j in 1:num_harmonic[param])
    else

        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, param]*cos(2 * π * j * (diff_T + t)/(num_harmonic[param] * 2)) +
                                                                            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, param]*sin(2 * π * j * (diff_T + t)/(num_harmonic[param] * 2)) for j in 1:num_harmonic[param])
    end
end

"""
# update_AR!(dict_hyperparams_and_fitted_components::Dict{String, Any}, order::Vector{Vector{Int64}}, param::Int64, t::Int64, s::Int64)

Updates the autoregressive (AR) component predictions for the specified parameter, time period, and scenario in the `dict_hyperparams_and_fitted_components` object.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `order::Vector{Vector{Int64}}`: A vector of vectors specifying the orders of the autoregressive process for each parameter.
- `param::Int64`: An integer indicating the index of the parameter for which the AR component is being updated.
- `t::Int64`: An integer indicating the time step at which the AR component is being updated.
- `s::Int64`: An integer indicating the scenario index.

## Returns
Updates the autoregressive (AR) component predictions for the specified parameter, time period, and scenario in the `dict_hyperparams_and_fitted_components` object.
"""
function update_AR!(dict_hyperparams_and_fitted_components::Dict{String, Any}, order::Vector{Vector{Int64}} , param::Int64, t::Int64, s::Int64)

    dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["ar"]["ϕ"][:, param][p] * dict_hyperparams_and_fitted_components["ar"]["value"][param, t - p, s] for p in order[param]) + 
                                                                                dict_hyperparams_and_fitted_components["ar"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
end

"""
# update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

Updates predictions for the specified parameter for a given time period and scenario in the `dict_hyperparams_and_fitted_components` object.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `param::Int64`: An integer indicating the index of the parameter for which the predictions are being updated.
- `t::Int64`: An integer indicating the time step at which the predictions are being updated.
- `s::Int64`: An integer indicating the scenario index.

## Returns
Updates predictions for the specified parameter for a given time period and scenario in the `dict_hyperparams_and_fitted_components` object.
"""
function update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

    dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                    dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] + 
                                                                    dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] +
                                                                    dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] +
                                                                    dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] 
                                                                                
end

"""
# update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, X_forecast::Matrix{Fl}, period_X::Int64, param::Int64, t::Int64, s::Int64) where Fl

Updates predictions for the specified parameter for a given time period and scenario in the `dict_hyperparams_and_fitted_components` object., incorporating the effects of explanatory variables for forecasting.

## Arguments
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `X_forecast::Matrix{Fl}`: A matrix of explanatory variables for forecasting.
- `period_X::Int64`: An integer indicating the period in the explanatory variables matrix `X_forecast` to be used for forecasting.
- `param::Int64`: An integer indicating the index of the parameter for which the predictions are being updated.
- `t::Int64`: An integer indicating the time step at which the predictions are being updated.
- `s::Int64`: An integer indicating the scenario index.

## Returns
Updates the parameter predictions for the specified parameter, time period, and scenario in the `dict_hyperparams_and_fitted_components` object, incorporating the effects of explanatory variables for forecasting.
"""
function update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, X_forecast::Matrix{Fl}, period_X::Int64, param::Int64, t::Int64, s::Int64) where Fl

    n_exp = size(X_forecast, 2)

    dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                    dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] + 
                                                                    dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] +
                                                                    dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] +
                                                                    dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] +
                                                                    sum(dict_hyperparams_and_fitted_components["explanatories"][j] * X_forecast[period_X, j] for j in 1:n_exp)
end

"""
# simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64)

Simulates future values of a time series using the specified GAS model and fitted components.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `y::Vector{Float64}`: The vector of observed values for the time series data.
- `steps_ahead::Int64`: An integer indicating the number of steps ahead to simulate.
- `num_scenarios::Int64`: An integer indicating the number of scenarios to simulate.

## Returns
- `pred_y::Matrix{Float64}`: A matrix containing the simulated values for the time series data. Each column represents a scenario, and each row represents a time step, including the fitted values and the specified number of steps ahead.
"""
function simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64)
    
    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    idx_params      = get_idxs_time_varying_params(time_varying_params) 
    order           = get_AR_order(ar)
    num_harmonic, _ = get_num_harmonic_and_seasonal_period(seasonality)
    dist_code       = get_dist_code(dist)

    T        = length(y)
    T_fitted = length(output.fit_in_sample)

    first_idx = T - T_fitted + 1

    ### PORQUE RECALCULAR O NUM_HARMONIC?
   
    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    else
        for i in idx_params
            if num_harmonic[i] == 0
                #println(num_harmonic[i])
                num_harmonic[i] = 1
            end
        end
        # length(idx_params) > length(num_harmonic) #considera os mesmos harmonicos para todos os parametros variantes, para não quebrar a update_S!
        # num_harmonic = Int64.(ones(length(idx_params)) * num_harmonic[1])
    end

    #println(num_harmonic)

    # if sum(vcat(order...)) == 0
    #     first_idx = 2
    # else 
    #     first_idx = maximum(vcat(order...)) + 1
    # end

    pred_y = zeros(T_fitted + steps_ahead, num_scenarios)
    pred_y[1:T_fitted, :] .= y[first_idx:end]

    Random.seed!(123)
    for t in 1:steps_ahead
        for s in 1:num_scenarios
            for i in idx_params
                update_score!(dict_hyperparams_and_fitted_components, pred_y, d, dist_code, i, T_fitted + t, s)
                if has_random_walk(random_walk, i)
                    update_rw!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_random_walk_slope(random_walk_slope, i)
                    update_rws!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_seasonality(seasonality, i)
                    update_S!(dict_hyperparams_and_fitted_components, num_harmonic, T - T_fitted, i, T_fitted + t, s)
                end
                if has_AR(ar, i)
                    update_AR!(dict_hyperparams_and_fitted_components, order, i, T_fitted + t, s)
                end
                update_params!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
            end
            pred_y[T_fitted + t, s] = sample_dist(dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s], dist)
        end
    end

    return pred_y
end

"""
# simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64) where {Fl}

Simulates future values of a time series using the specified GAS model, fitted components, and explanatory variables for forecasting.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `dict_hyperparams_and_fitted_components::Dict{String, Any}`: A dictionary containing hyperparameters, fitted components, and other relevant information.
- `y::Vector{Float64}`: The vector of observed values for the time series data.
- `X_forecast::Matrix{Fl}`: The matrix of explanatory variables for forecasting.
- `steps_ahead::Int64`: An integer indicating the number of steps ahead to simulate.
- `num_scenarios::Int64`: An integer indicating the number of scenarios to simulate.

## Returns
- `pred_y::Matrix{Float64}`: A matrix containing the simulated values for the time series data. Each column represents a scenario, and each row represents a time step, including the fitted values and the specified number of steps ahead.
"""
function simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64) where {Fl}
    
    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    idx_params      = get_idxs_time_varying_params(time_varying_params) 
    order           = get_AR_order(ar)
    num_harmonic, _ = get_num_harmonic_and_seasonal_period(seasonality)
    dist_code       = get_dist_code(dist)

    T        = length(y)
    T_fitted = length(output.fit_in_sample)

    first_idx = T - T_fitted + 1

    ### PORQUE RECALCULAR O NUM_HARMONIC?
    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    else
        for i in idx_params
            if num_harmonic[i] == 0
                #println(num_harmonic[i])
                num_harmonic[i] = 1
            end
        end
    end

    # if sum(vcat(order...)) == 0
    #     first_idx = 2
    # else 
    #     first_idx = maximum(vcat(order...)) + 1
    # end

    pred_y = zeros(T_fitted + steps_ahead, num_scenarios)
    pred_y[1:T_fitted, :] .= y[first_idx:end]

    Random.seed!(123)
    for t in 1:steps_ahead
        for s in 1:num_scenarios
            for i in idx_params
                update_score!(dict_hyperparams_and_fitted_components, pred_y, d, dist_code, i, T_fitted + t, s)
                if has_random_walk(random_walk, i)
                    update_rw!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_random_walk_slope(random_walk_slope, i)
                    update_rws!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_seasonality(seasonality, i)
                    update_S!(dict_hyperparams_and_fitted_components, num_harmonic, T - T_fitted, i, T_fitted + t, s)
                end
                if has_AR(ar, i)
                    update_AR!(dict_hyperparams_and_fitted_components, order, i, T_fitted + t, s)
                end
                update_params!(dict_hyperparams_and_fitted_components, X_forecast, t, i, T_fitted + t, s)
            end
            pred_y[T_fitted + t, s] = sample_dist(dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s], dist)
        end
    end

    return pred_y
end

"""
# get_mean_and_intervals_prediction(pred_y::Matrix{Fl}, steps_ahead::Int64, probabilistic_intervals::Vector{Float64}) where Fl

Calculates the mean and intervals of predictions from simulated scenarios.

## Arguments
- `pred_y::Matrix{Fl}`: A matrix containing simulated values for the time series data. Each column represents a scenario, and each row represents a time step, including the fitted values and the specified number of steps ahead.
- `steps_ahead::Int64`: An integer indicating the number of steps ahead for the predictions.
- `probabilistic_intervals::Vector{Float64}`: A vector of floats indicating the probabilistic intervals to be computed.

## Returns
- `dict_forec::Dict{String, Any}`: A dictionary containing the mean prediction and intervals for the simulated scenarios. It has the following structure:
  - `mean`: A vector containing the mean prediction for each time step.
  - `scenarios`: A matrix containing the simulated scenarios, where each column represents a scenario.
  - `intervals`: A dictionary containing the upper and lower bounds of the interval for each probabilistic level. Each probabilistic level is represented by a string key indicating the percentage, and its value is another dictionary with the following structure:
    - `upper`: A vector containing the interval's upper bound for each time step.
    - `lower`: A vector containing the interval's  lower bound for each time step.
"""
function get_mean_and_intervals_prediction(pred_y::Matrix{Fl}, steps_ahead::Int64, probabilistic_intervals::Vector{Float64}) where Fl
    
    forec           = zeros(steps_ahead)
    forec_intervals = zeros(steps_ahead, length(probabilistic_intervals) * 2)

    dict_forec = Dict{String, Any}()
    dict_forec["intervals"] = Dict{String, Any}()

    for t in 1:steps_ahead
        forec[t] = mean(pred_y[end - steps_ahead + 1:end, :][t, :])

        for q in eachindex(probabilistic_intervals)
            α = round(1 - probabilistic_intervals[q], digits = 4)
            forec_intervals[t, 2 * q - 1] = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], α/2)
            forec_intervals[t, 2 * q]     = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], 1 - α/2)
        end
    end

    for q in eachindex(probabilistic_intervals)
        α = Int64(100 * probabilistic_intervals[q])
        dict_forec["intervals"]["$α"] = Dict{String, Any}()
        dict_forec["intervals"]["$α"]["upper"] = forec_intervals[:, 2 * q]
        dict_forec["intervals"]["$α"]["lower"] = forec_intervals[:, 2 * q - 1]
    end

    dict_forec["mean"] = forec
    dict_forec["scenarios"] = pred_y[end - steps_ahead + 1:end, :]

    return dict_forec
end

"""
# predict(gas_model::GASModel, output::Output, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95])

Predicts future values of a time series using the specified GAS model.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `y::Vector{Float64}`: The vector of observed values for the time series data.
- `steps_ahead::Int64`: An integer indicating the number of steps ahead to predict.
- `num_scenarios::Int64`: An integer indicating the number of scenarios to simulate.
- `probabilistic_intervals::Vector{Float64}`: A vector of floats indicating the probabilistic intervals to be computed. Default is `[0.8, 0.95]`.

## Returns
- `dict_forec::Dict{String, Any}`: A dictionary containing the mean prediction and intervals for the simulated scenarios. It has the following structure:
  - `mean`: A vector containing the mean prediction for each time step.
  - `scenarios`: A matrix containing the simulated scenarios, where each column represents a scenario.
  - `intervals`: A dictionary containing the upper and lower bounds of the interval for each probabilistic level. Each probabilistic level is represented by a string key indicating the percentage, and its value is another dictionary with the following structure:
    - `upper`: A vector containing the interval's upper bound for each time step.
    - `lower`: A vector containing the interval's  lower bound for each time step.
"""
function predict(gas_model::GASModel, output::Output, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95])
    
    new_output = deepcopy(output)
    if typeof(gas_model.dist) == LogNormalDistribution
        new_output.fitted_params = convert_to_log_scale(new_output.fitted_params)
        y = log.(y)
    end

    dict_hyperparams_and_fitted_components = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, new_output, steps_ahead, num_scenarios)
    pred_y                                 = simulate(gas_model, new_output, dict_hyperparams_and_fitted_components, y, steps_ahead, num_scenarios)

    dict_forec = get_mean_and_intervals_prediction(pred_y, steps_ahead, probabilistic_intervals)

    if typeof(gas_model.dist) == LogNormalDistribution
        dict_forec = convert_forecast_to_exp_scale(dict_forec)
    end

    return dict_forec
end

"""
# predict(gas_model::GASModel, output::Output, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95]) where {Ml, Fl}

Predicts future values of a time series using the specified GAS model and explanatory variables for forecasting.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `y::Vector{Float64}`: The vector of observed values for the time series data.
- `X_forecast::Matrix{Fl}`: The matrix of explanatory variables for forecasting.
- `steps_ahead::Int64`: An integer indicating the number of steps ahead to predict.
- `num_scenarios::Int64`: An integer indicating the number of scenarios to simulate.
- `probabilistic_intervals::Vector{Float64}`: A vector of floats indicating the probabilistic intervals to be computed. Default is `[0.8, 0.95]`.

## Returns
- `dict_forec::Dict{String, Any}`: A dictionary containing the mean prediction and intervals for the simulated scenarios. It has the following structure:
  - `mean`: A vector containing the mean prediction for each time step.
  - `scenarios`: A matrix containing the simulated scenarios, where each column represents a scenario.
  - `intervals`: A dictionary containing the upper and lower bounds of the interval for each probabilistic level. Each probabilistic level is represented by a string key indicating the percentage, and its value is another dictionary with the following structure:
    - `upper`: A vector containing the interval's upper bound for each time step.
    - `lower`: A vector containing the interval's  lower bound for each time step.
"""
function predict(gas_model::GASModel, output::Output, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95]) where {Ml, Fl}
    
    new_output = deepcopy(output)

    if typeof(gas_model.dist) == LogNormalDistribution
        new_output.fitted_params = convert_to_log_scale(new_output.fitted_params)
        y = log.(y)
        #X_forecast = log.(X_forecast) #Devo fazer isso?
    end

    dict_hyperparams_and_fitted_components = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, new_output, X_forecast, steps_ahead, num_scenarios)
    pred_y                                 = simulate(gas_model, new_output, dict_hyperparams_and_fitted_components, y, X_forecast, steps_ahead, num_scenarios)

    dict_forec = get_mean_and_intervals_prediction(pred_y, steps_ahead, probabilistic_intervals)

    if typeof(gas_model.dist) == LogNormalDistribution
        dict_forec = convert_forecast_to_exp_scale(dict_forec)
    end

    return dict_forec
end
