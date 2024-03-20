"""
# get_idxs_time_varying_params(time_varying_params::Vector{Bool})

Returns the indices of time-varying parameters.

## Arguments
- `time_varying_params::Vector{Bool}`: A vector indicating whether each parameter is time-varying (`true`) or fixed (`false`).

## Returns
- `idxs::Vector{Int}`: A vector containing the indices of time-varying parameters.
"""
get_idxs_time_varying_params(time_varying_params::Vector{Bool}) = findall(i -> i == true, time_varying_params)

"""
# get_fitted_values(gas_model::GASModel, model::Ml, X::Union{Missing, Matrix{Fl}})

Returns the fitted values and components of the specified GAS model.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `model::Ml`: The optimization model with fitted parameters and components.
- `X::Union{Missing, Matrix{Fl}}`: Explanatory variables used in the model. Set to `missing` if not provided.

## Returns
- `fit_in_sample::Vector{Fl}`: A vector containing the fitted values in the sample.
- `fitted_params::Dict{String, Vector{Float64}}`: A dictionary containing the fitted parameters for each time-varying component. Each key represents a parameter, and its value is a vector of fitted values.
- `components::Dict{String, Any}`: A dictionary containing the components of the model. Each key represents a parameter, and its value is another dictionary containing the components' details.
"""
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

"""
# get_std_residuals(y::Vector{Fl}, fit_in_sample::Vector{Fl})

Calculates the standardized residuals of a time series' model.

## Arguments
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `fit_in_sample::Vector{Fl}`: The vector of fitted values in the sample.

## Returns
- `std_res::Vector{Fl}`: A vector containing the standardized residuals.
"""
function get_std_residuals(y::Vector{Fl}, fit_in_sample::Vector{Fl}) where Fl
    residuals = y .- fit_in_sample
    std_res = (residuals .- mean(residuals)) / std(residuals)

    return std_res
end

"""
# get_cs_residuals(y::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64)

Calculates the conditional score (CS) residuals for a time series based on the fitted parameters of a GAS model.

## Arguments
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `fitted_params::Dict{String, Vector{Float64}}`: A dictionary containing the fitted parameters of the GAS model. Each key represents a parameter name, and the corresponding value is a vector of fitted values for each time period.
- `dist_code::Int64`: An integer representing the code of the distribution used in the GAS model.

## Returns
- `cs_residuals::Matrix{Fl}`: A matrix containing the conditional score (CS) residuals. Each row corresponds to a time period, and each column represents a different parameter.
"""
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

"""
# get_quantile_residuals(y::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64)

Calculates the quantile residuals for a time series based on the fitted parameters of a GAS model.

## Arguments
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `fitted_params::Dict{String, Vector{Float64}}`: A dictionary containing the fitted parameters of the GAS model. Each key represents a parameter name, and the corresponding value is a vector of fitted values for each time period.
- `dist_code::Int64`: An integer representing the code of the distribution used in the GAS model.

## Returns
- `q_residuals::Vector{Fl}`: A vector containing the quantile residuals for each time period.
"""
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

"""
# get_residuals(y::Vector{Float64}, fit_in_sample::Vector{Float64}, fitted_params::Dict{String, Vector{Float64}}, dist_code::Int64)

Calculates various types of residuals for a time series based on the fitted parameters of a GAS model.

## Arguments
- `y::Vector{Float64}`: The vector of observed values for the time series data.
- `fit_in_sample::Vector{Float64}`: The vector of fitted values obtained in the in-sample fitting process.
- `fitted_params::Dict{String, Vector{Float64}}`: A dictionary containing the fitted parameters of the GAS model. Each key represents a parameter name, and the corresponding value is a vector of fitted values for each time period.
- `dist_code::Int64`: An integer representing the code of the distribution used in the GAS model.

## Returns
- `dict_residuals::Dict{String, Union{Vector{Float64}, Matrix{Float64}}}`: A dictionary containing various types of residuals.
  - `std_residuals`: A vector of standardized residuals.
  - `cs_residuals`: A matrix of conditional score residuals.
  - `q_residuals`: A vector of quantile residuals.
"""
function get_residuals(y::Vector{Float64}, fit_in_sample::Vector{Float64}, fitted_params::Dict{String, Vector{Float64}}, dist::ScoreDrivenDistribution)

    # Getting dist
    dist == LogNormalDistribution ? dist_code = get_dist_code(NormalDistribution) : dist_code = get_dist_code(dist)

    # Getting std residuals
    std_res = get_std_residuals(y, fit_in_sample)

    # Getting Conditional Score Residuals 
    cs_residuals = get_cs_residuals(y, fitted_params, dist_code)

    # Getting Quantile Residuals
    if dist == LogNormalDistribution
        q_residuals = get_quantile_residuals(log.(y), fitted_params, dist_code)
    else
        q_residuals = get_quantile_residuals(y, fitted_params, dist_code)
    end
    
    dict_residuals = Dict{String, Union{Vector{Float64}, Matrix{Float64}}}()
    dict_residuals["std_residuals"] = std_res
    dict_residuals["cs_residuals"]  = cs_residuals
    dict_residuals["q_residuals"]   = q_residuals

    return dict_residuals
    
end

"""
# aic(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

Calculates the Akaike Information Criterion (AIC) for a given GAS model.

## Arguments
- `model::Ml`: The optimization model used for fitting the GAS model.
- `parameters::Matrix{Gl}`: A matrix containing the estimated parameters of the GAS model. Each row represents a time period, and each column represents a parameter.
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `dist::ScoreDrivenDistribution`: The distribution used in the GAS model.

## Returns
- `aic::Gl`: The Akaike Information Criterion (AIC) value.
"""
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

"""
# bic(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

Calculates the Bayesian Information Criterion (BIC) for a given GAS model.

## Arguments
- `model::Ml`: The optimization model used for fitting the GAS model.
- `parameters::Matrix{Gl}`: A matrix containing the estimated parameters of the GAS model. Each row represents a time period, and each column represents a parameter.
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `dist::ScoreDrivenDistribution`: The distribution used in the GAS model.

## Returns
- `bic::Gl`: The Bayesian Information Criterion (BIC) value.
"""
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

"""
# aicc(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

Calculates the corrected Akaike Information Criterion (AICc) for a given GAS model.

## Arguments
- `model::Ml`: The optimization model used for fitting the GAS model.
- `parameters::Matrix{Gl}`: A matrix containing the estimated parameters of the GAS model. Each row represents a time period, and each column represents a parameter.
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `dist::ScoreDrivenDistribution`: The distribution used in the GAS model.

## Returns
- `aicc::Gl`: The corrected Akaike Information Criterion (AICc) value.
"""
function aicc(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

    #num_params = get_num_params(dist)
    
    num_parameters = length(all_variables(model))

    AIC = aic(model, parameters, y, dist)

    return AIC + (2 *  num_parameters * ( num_parameters + 1)) / (length(y[2:end]) -  num_parameters - 1)
end

"""
# get_information_criteria(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

Calculates various information criteria for assessing the fit of a GAS model.

## Arguments
- `model::Ml`: The optimization model used for fitting the GAS model.
- `parameters::Matrix{Gl}`: A matrix containing the estimated parameters of the GAS model. Each row represents a time period, and each column represents a parameter.
- `y::Vector{Fl}`: The vector of observed values for the time series data.
- `dist::ScoreDrivenDistribution`: The distribution used in the GAS model.

## Returns
- `dict::Dict{String, Float64}`: A dictionary containing the calculated information criteria.
  - `"aic"`: Akaike Information Criterion (AIC) value.
  - `"bic"`: Bayesian Information Criterion (BIC) value.
  - `"aicc"`: Corrected Akaike Information Criterion (AICc) value.
"""
function get_information_criteria(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, dist::ScoreDrivenDistribution) where {Ml, Fl, Gl}

    dict = Dict{String, Float64}()

    dict["aic"]  = aic(model, parameters, y, dist)
    dict["bic"]  = bic(model, parameters, y, dist)
    dict["aicc"] = aicc(model, parameters, y, dist)

    return dict
end

"""
# is_valid_model(output::Output) -> Bool

Checks whether the output of a GAS model represents an optimal model or if it indicates an invalid model or a numeric error.

## Arguments
- `output::Output`: The output structure containing information about the fitted model.

## Returns
- `valid::Bool`: A boolean indicating whether the model is valid (`true`) or not (`false`).
"""
function is_valid_model(output::Output)

    return output.model_status ∉ ["INVALID_MODEL", "NUMERIC_ERROR", "TIME_LIMIT"]
end

"""
# fit_AR_model(y::Vector{Fl}, order::Union{Vector{Int64}, Vector{Nothing}})

Fits an autoregressive (AR) model to the provided time series data.

## Arguments
- `y::Vector{Fl}`: A vector containing the observed values for the time series data.
- `order::Union{Vector{Int64}, Vector{Nothing}}`: A vector specifying the order of the AR model.

## Returns
- `y_hat::Vector{Fl}`: The fitted values generated by the AR model.
- `ϕ::Vector{Fl}`: The estimated coefficients of the AR model.
- `c::Fl`: The intercept term of the AR model.
"""
function fit_AR_model(y::Vector{Fl}, order::Union{Vector{Int64}, Vector{Nothing}}) where Fl

    T         = length(y)
    max_order = maximum(order) 

    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    set_optimizer_attribute(model, "print_level", 0)

    @variable(model, c)
    @variable(model, ϕ[order])
    @variable(model, y_hat[1:T])

    # println(order)
    # println(typeof(order))
    @constraint(model, [t = max_order+1:T], y_hat[t] == c + sum(ϕ[i]*y[t - i] for i in order))
   
    @objective(model, Min, sum((y .- y_hat).^2))
    optimize!(model)

    return JuMP.value.(y_hat), JuMP.value.(ϕ), JuMP.value(c)
end

"""
# fit_harmonics(y::Vector{Fl}, seasonal_period::Int64, stochastic::Bool) -> Tuple{Vector{Fl}, Vector{Fl}}

Fits harmonic components to the provided time series data to capture seasonality.

## Arguments
- `y::Vector{Fl}`: A vector containing the observed values for the time series data.
- `seasonal_period::Int64`: An integer representing the seasonal period of the data.
- `stochastic::Bool`: A boolean indicating whether the model is stochastic.

## Returns
- `γ::Vector{Fl}`: The estimated coefficients of the cosine terms in the harmonic model.
- `γ_star::Vector{Fl}`: The estimated coefficients of the sine terms in the harmonic model.
"""
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

