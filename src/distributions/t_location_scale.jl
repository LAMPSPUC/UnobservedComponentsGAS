"""
mutable struct tLocationScaleDistribution

    A mutable struct for representing the tLocationScale distribution.

    # Fields
    - `μ::Union{Missing, Float64}`: Mean parameter.
    - `σ²::Union{Missing, Float64}`: Variance parameter.
    - `ν::Union{Missing, Int64}`: Degrees of freedom parameter.
"""
mutable struct tLocationScaleDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
    ν::Union{Missing, Int64}
end

"""
tLocationScaleDistribution()

Outer constructor for the tLocationScale distribution, with no arguments specified.
    
    # Returns
    - The tLocationScaleDistribution struct with both fields set to missing.
"""
function tLocationScaleDistribution()
    return tLocationScaleDistribution(missing, missing, missing)
end

"""
score_tlocationscale(μ, σ², ν,  y) 

# Compute the score vector of the Student's t-location-scale distribution, considering the specified parameters and observation.
    
    # Arguments

    - `μ`: The value of the location parameter.
    - `σ²`: The value of the scale parameter.
    - `ν`: The degrees of freedom parameter.
    - `y`: The value of the observation.
    
    # Returns
    - A vector of type Float64, where the first element corresponds to the score related to the location parameter, and the second element corresponds to the score related to the scale parameter.
    
"""
function score_tlocationscale(μ, σ², ν,  y) 
  
    return [((ν + 1) * (y - μ)) / ((y - μ)^2 + σ² * ν), -(ν * (σ² - (y - μ)^2)) / (2 * σ² * (ν * σ² + (y - μ)^2))]
end

"""
fisher_information_tlocationscale(μ, σ², ν)

# Compute the Fisher information matrix of the Student's t-location-scale distribution, considering the specified parameters.
    
# Arguments

- `μ`: The value of the location parameter.
- `σ²`: The value of the scale parameter.
- `ν`: The degrees of freedom parameter.

# Returns
- A 2x2 matrix of type Float64 representing the Fisher information matrix. 
  The diagonal elements correspond to the Fisher information related to the location parameter and the scale parameter, respectively.
  Off-diagonal elements are zero.
"""
function fisher_information_tlocationscale(μ, σ², ν)
    return [((ν + 1.0)/(σ² * (ν + 3.0))) 0.0 ; 0.0 (ν / ((2 * σ²^2) * (ν + 3.0)))]
end

"""
logpdf_tlocationscale(μ, σ², ν, y)

# Compute the logarithm of the probability density function (PDF) of the Student's t-location-scale distribution, considering the specified parameters and observation.
    
    # Arguments

    - `μ`: The value of the location parameter.
    - `σ²`: The value of the scale parameter.
    - `ν`: The degrees of freedom parameter.
    - `y`: The value of the observation.
    
    # Returns
    - The logarithm of the probability density function (PDF) of the Student's t-location-scale distribution evaluated at the specified observation.
    
"""
function logpdf_tlocationscale(μ, σ², ν, y)

    return logpdf_tlocationscale([μ, σ², ν], y)
end

"""
logpdf_tlocationscale(param, y)

# Compute the logarithm of the probability density function (PDF) of the Student's t-location-scale distribution, considering the specified parameters and observation.
    
    # Arguments

    - `param`: A vector containing the parameters of the distribution in the following order: [location parameter, scale parameter, degrees of freedom parameter].
    - `y`: The value of the observation.
    
    # Returns
    - The logarithm of the probability density function (PDF) of the Student's t-location-scale distribution evaluated at the specified observation, using the provided parameters.
    
"""
function logpdf_tlocationscale(param, y)

    if param[2] < 0
        param[2] = 1e-4
    end

    if param[3] < 0
        param[3] = 3
    end

    return log(gamma((param[3] + 1)/2)) - log(gamma(param[3]/2)) - (1/2)*log(π*param[3]*param[2]) - (param[3] + 1)/2 * log(1 + ((y - param[1])^2)/(param[3]*param[2]))
end

"""
# Compute the cumulative distribution function (CDF) of the Student's t-location-scale distribution, given the specified parameters and observation.
    
# Arguments

- `param`: A vector containing the distribution parameters in the following order: [location parameter, scale parameter, degrees of freedom parameter].
- `y`: The observed value.

# Returns
- The cumulative probability up to `y` according to the Student's t-location-scale distribution with the provided parameters.

"""
function cdf_tlocationscale(param::Vector{Float64}, y::Fl) where Fl

    return Distributions.cdf(TDist(param[3]), (y - param[1]) / sqrt(param[2]))
end

"""
get_dist_code(dist::tLocationScaleDistribution)

# Get the distribution code corresponding to the tLocationScale distribution.
    
# Arguments

- `dist`: The tLocationScaleDistribution object.

# Returns
- The distribution code corresponding to the tLocationScale distribution. (In this case, it always returns 2.)

"""
function get_dist_code(dist::tLocationScaleDistribution)
    return 2
end

"""
get_num_params(dist::tLocationScaleDistribution)

# Get the number of parameters associated with the tLocationScaleDistribution.
    
# Arguments

- `dist`: The tLocationScaleDistribution object.

# Returns
- The number of parameters associated with the tLocationScaleDistribution. (In this case, it always returns 3.)

"""
function get_num_params(dist::tLocationScaleDistribution)
    return 3
end

"""
sample_dist(param::Vector{Fl}, dist::tLocationScaleDistribution) where Fl
    
    # Sample from the tLocationScaleDistribution using the provided parameters.
    
    # Arguments

    - `param`: A vector containing the distribution parameters in the following order: [location parameter, scale parameter, degrees of freedom parameter].
    - `dist`: The tLocationScaleDistribution object.
    
    # Returns
    - A random sample drawn from the tLocationScaleDistribution with the provided parameters.
    
"""
function sample_dist(param::Vector{Fl}, dist::tLocationScaleDistribution) where Fl
    
    if param[2] < 0
        param[2] = 1e-4
    end

    return param[1] + sqrt(param[2]) * rand(TDist(param[3]), 1)[1]
end

"""
find_ν(y::Vector{Fl}, dist::tLocationScaleDistribution) where Fl

    # Estimate the degrees of freedom parameter (ν) for the tLocationScaleDistribution given the observed data.
    
    # Arguments

    - `y`: A vector of observations.
    - `dist`: The tLocationScaleDistribution object.
    
    # Returns
    - The estimated degrees of freedom parameter (ν).
"""
function find_ν(y::Vector{Fl}, dist::tLocationScaleDistribution) where Fl

    T = length(y)
    dist_name = DICT_CODE[get_dist_code(dist)]

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", 100000)
    set_silent(model)

    @variable(model, μ)
    @variable(model, σ² ≥ 1e-4)
    @variable(model, ν >= 3.1)

    register(model, :log_pdf, 4, logpdf_tlocationscale; autodiff = true)
    @NLobjective(model, Max, sum(log_pdf(μ, σ², ν, y[t]) for t in 2:T))

    set_start_value(μ, mean(y))
    set_start_value(σ², var(diff(y)))
    set_start_value(ν, 10)

    optimize!(model)

    return Int64(round(value(ν)))
end

"""
check_positive_constrainst(dist::tLocationScaleDistribution)

 Indicates which parameter of the t-LocationScale distribution must be positive.
    
    # Arguments

    - `dist::tLocationScaleDistribution`: The structure that represents the t-LocationScale distribution.
    
    # Returns
    - A boolean vector indicating that the scale and degrees of freedom parameters must be positive, while the location parameter does not necessarily need to be positive.
"""
function check_positive_constrainst(dist::tLocationScaleDistribution)
    return [false, true, true]
end

"""
get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::tLocationScaleDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl

    # Compute initial parameters for the tLocationScaleDistribution model based on the provided data and settings.
    
    # Arguments

    - `y`: A vector of observations.
    - `time_varying_params`: A boolean vector indicating whether each parameter is time-varying.
    - `dist`: The tLocationScaleDistribution object.
    - `seasonality`: A dictionary indicating the presence of seasonal components in the time-varying parameter's dynamic.
    
    # Returns
    - initial_params:Dict{Int64, Any}: A dictionary containing the initial values for each parameter of the t-LocationScal distribution model.
        - 1: Initial values for the location parameter, which can be fixed or time-varying.
        - 2: Initial values for the scale parameter, which can be fixed or time-varying.
        - 3: Initial values for the degrees of freedom parameter, which can be fixed or time-varying.
"""
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::tLocationScaleDistribution, seasonality::Dict{Int64, Union{Bool, Int64}}) where Fl

    T         = length(y)
    dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict{Int64, Any}()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = var(diff(y))*ones(T) #get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y ,ones(T) * var(diff(y)), (y.^2) ./ (ones(T) * var(diff(y))) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = var(diff(y)) #* (initial_params[3] - 2) / initial_params[3]
    end

    if time_varying_params[3]
        initial_params[3] = (y.^2) ./ (ones(T) * var(diff(y))) # Estimador para ν pelo metodo dos momentos
    else
        initial_params[3] = T - 1
    end

    return initial_params
end

"""
get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::tLocationScaleDistribution) where Fl

    # Compute seasonal variances for the tLocationScaleDistribution model based on the provided data.
    
    # Arguments

    - `y::Vector{Fl}`: A vector of observations.
    - `seasonal_period::Int64`: The length of the seasonal period.
    - `dist::tLocationScaleDistribution`: The tLocationScaleDistribution object.
    
    # Returns
    - seasonal_variances::Vector{Fl}: A vector containing the seasonal variances computed based on the provided data.
    
"""
function get_seasonal_var(y::Vector{Fl},seasonal_period::Int64, dist::tLocationScaleDistribution) where Fl
    num_periods = ceil(Int, length(y) / seasonal_period)
    seasonal_variances = zeros(Fl, length(y))

    for i in 1:seasonal_period
        month_data = y[i:seasonal_period:end]
        num_observations = length(month_data)
        if num_observations > 0
            variance = Distributions.fit(Normal, month_data).σ^2
            
            for j in 1:num_observations
                seasonal_variances[i + (j - 1) * seasonal_period] = variance
            end
        end
    end
    return seasonal_variances 
end
 

"""
find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}; 
                                          α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl}

    # Finds the optimal initial model for local search by considering two possible values for ν, using the GAS model and the provided data.
    
    # Arguments

    - `gas_model::GASModel`: The GAS model.
    - `y::Vector{Fl}`: A vector of observations.
    - `α::Float64`: (Optional) The alpha value for the GAS model.
    - `robust_prop::Float64`: (Optional) The proportion of data to use for robust fitting.
    - `number_max_iterations::Int64`: (Optional) The maximum number of iterations for optimization.
    - `max_optimization_time::Float64`: (Optional) The maximum time allowed for optimization.
    - `initial_values::Union{Dict{String, Any}, Missing}`: (Optional) Initial parameter values for the optimization process.
    
    # Returns
    - best_model: The most accurate estimated model, determined through comparison of information criteria (AICc).
    - best_ν: The value of the degrees of freedom parameter employed in the best model.
    
"""
function find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}; 
                                          α::Float64 = 0.5, robust::Bool = false, robust_prop::Float64 = 0.7, 
                                          κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    T    = length(y)
    dist = gas_model.dist

    optimal_ν = find_ν(y, dist)
    heuristic_ν  = T - 1

    opt_model, opt_parameters, initial_values = create_model(gas_model, y, optimal_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                             κ_min = κ_min, κ_max = κ_max)

    optimal_model = fit(gas_model, y, opt_model, opt_parameters, initial_values; α = α, robust_prop = robust_prop)

    heu_model, heu_parameters, initial_values = create_model(gas_model, y, heuristic_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                             κ_min = κ_min, κ_max = κ_max)

    heuristic_model = fit(gas_model, y, heu_model, heu_parameters, initial_values; α = α, robust_prop = robust_prop)

    if !is_valid_model(optimal_model) || optimal_model.information_criteria["aicc"] > heuristic_model.information_criteria["aicc"]
        @info("Considering ν  = T-1")
        best_model = heuristic_model
        best_ν = heuristic_ν

    elseif !is_valid_model(heuristic_model) || heuristic_model.information_criteria["aicc"] ≥ optimal_model.information_criteria["aicc"]
        @info("Considering ν provided by optimization")
        best_model = optimal_model
        best_ν = optimal_ν
    end

    return best_model, best_ν
end 

"""
find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; 
                                          α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    # Finds the optimal initial model for local search by considering two possible values for ν, using the GAS model and the provided data and exogenous variables.
    
    # Arguments

    - `gas_model::GASModel`: The GAS model object.
    - `y::Vector{Fl}`: A vector of observations.
    - `X::Matrix{Fl}`: A matrix of exogenous variables.
    - `α::Float64`: (Optional) The alpha value for the GAS model.
    - `robust_prop::Float64`: (Optional) The proportion of data to use for robust fitting.
    - `number_max_iterations::Int64`: (Optional) The maximum number of iterations for optimization.
    - `max_optimization_time::Float64`: (Optional) The maximum time allowed for optimization.
    - `initial_values::Union{Dict{String, Any}, Missing}`: (Optional) Initial parameter values for the optimization process.
    
    # Returns
    - best_model: The most accurate estimated model, determined through comparison of information criteria (AICc).
    - best_ν: The value of the degrees of freedom parameter employed in the best model.

"""
function find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; 
                                          α::Float64 = 0.5, robust::Bool = false, robust_prop::Float64 = 0.7, 
                                          κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}
    T    = length(y)
    dist = gas_model.dist

    optimal_ν = find_ν(y, dist)
    heuristic_ν  = T - 1

    opt_model, opt_parameters, initial_values = create_model(gas_model, y, X, optimal_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                             κ_min = κ_min, κ_max = κ_max)

    optimal_model = fit(gas_model, y, X, opt_model, opt_parameters, initial_values; α = α, robust = robust, robust_prop = robust_prop)

    heu_model, heu_parameters, initial_values = create_model(gas_model, y, X, heuristic_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                             κ_min = κ_min, κ_max = κ_max)

    heuristic_model = fit(gas_model, y, X, heu_model, heu_parameters, initial_values; α = α, robust = robust, robust_prop = robust_prop)

    if !is_valid_model(optimal_model) || optimal_model.information_criteria["aicc"] > heuristic_model.information_criteria["aicc"]
        @info("Considering ν  = T-1")
        best_model = heuristic_model
        best_ν = heuristic_ν

    elseif !is_valid_model(heuristic_model) || heuristic_model.information_criteria["aicc"] ≥ optimal_model.information_criteria["aicc"]
        @info("Considering ν provided by optimization")
        best_model = optimal_model
        best_ν = optimal_ν
    end

    return best_model, best_ν
end

"""
fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl};
                                            tol::Float64 = 0.01, α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                            number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                            initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    # Fits a t-location-scale distribution using local search, aiming to optimize the degrees of freedom parameter (ν) by comparing information criteria (AICc).
    # Arguments

    - `gas_model::GASModel`: The GAS model object.
    - `y::Vector{Fl}`: A vector of observations.
    - `tol::Float64`: (Optional) The tolerance level for stopping the local search.
    - `α::Float64`: (Optional) The regularization value for the GAS model.
    - `robust_prop::Float64`: (Optional) The proportion of data to use for robust fitting.
    - `number_max_iterations::Int64`: (Optional) The maximum number of iterations for optimization.
    - `max_optimization_time::Float64`: (Optional) The maximum time allowed for optimization.
    - `initial_values::Union{Dict{String, Any}, Missing}`: (Optional) Initial parameter values for the optimization process.
    
    # Returns
    - best_model: The best-fitted model based on the t-location-scale distribution after the local search is determined by comparing information criteria (AICc).
"""
function fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl};
                                            tol::Float64 = 0.01, α::Float64 = 0.5, robust::Bool = false, robust_prop::Float64 = 0.7,
                                            κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, number_max_iterations::Int64 = 30000, 
                                            max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    T    = length(y)
    dist = gas_model.dist

    fitted_model_ν, first_ν = find_first_model_for_local_search(gas_model, y;  α = α, robust = robust, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                                κ_min = κ_min, κ_max = κ_max, max_optimization_time =  max_optimization_time, initial_values = initial_values)

    model_lower, parameters_lower, initial_values_lower = create_model(gas_model, y,  first_ν-1; number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                                 κ_min = κ_min, κ_max = κ_max)

    fitted_model_ν_lower = fit(gas_model, y, model_lower, parameters_lower, initial_values_lower; α = α, robust = robust, robust_prop = robust_prop)
    
    model_upper, parameters_upper, initial_values_upper = create_model(gas_model, y, first_ν+1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                                 κ_min = κ_min, κ_max = κ_max)

    fitted_model_ν_upper = fit(gas_model, y, model_upper, parameters_upper, initial_values_upper; α = α,robust = robust, robust_prop = robust_prop)

    aicc_ = [is_valid_model(fitted_model_ν_lower) ? fitted_model_ν_lower.information_criteria["aicc"] : Inf,
            fitted_model_ν.information_criteria["aicc"],
            is_valid_model(fitted_model_ν_upper) ? fitted_model_ν_upper.information_criteria["aicc"] : Inf]

    aicc_[findall(i -> i == -Inf, aicc_)] .= Inf

    #historic_aicc = hcat([first_ν-1, first_ν, first_ν + 1],  aicc_)

    if argmin(aicc_) == 2
        @info("It was not necessary to do the local search!")
        #push!(historic_aicc, aicc_[2])
        best_model = fitted_model_ν
    else
        @info("Starting the local search!")
        search = true
        if argmin(aicc_) == 1
            factor     = -1
            current_ν  = first_ν - 1
            best_model = fitted_model_ν_lower
            best_aicc  = aicc_[1]

            @info("Current ν : $current_ν")
            @info("Best AICc: $best_aicc")
        else
            factor     = 1
            current_ν  = first_ν + 1
            best_model = fitted_model_ν_upper
            best_aicc  = aicc_[3]

            @info("Current ν : $current_ν")
            @info("Best AICc: $best_aicc")
        end

        #push!(historic_aicc, best_aicc)

        while search && current_ν > 3.0
            current_ν += 1 * factor
            @info("Trying ν = $current_ν")
            initial_values = create_output_initialization_from_fit(best_model, gas_model)

            model, parameter, _ = create_model(gas_model, y, current_ν;  number_max_iterations = number_max_iterations,
                                            max_optimization_time =  max_optimization_time, initial_values = initial_values)

            fitted_model   = fit(gas_model, y, model, parameter, initial_values; α = α,robust = robust,  robust_prop = robust_prop)

            if fitted_model.information_criteria["aicc"] < best_aicc#(fitted_model.information_criteria["aicc"] - best_aicc) / best_aicc < -tol && is_valid_model(fitted_model)
                
                best_model = fitted_model
                best_aicc  = fitted_model.information_criteria["aicc"]
               
                @info("Best AICc: $best_aicc")
            else 
                search = false
            end
        end
    end

    return best_model#, historic_aicc
end

"""
fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl};
                                         tol::Float64 = 0.01, α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                         number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl}

Fits a t-location-scale distribution with exogenous variables using local search, aiming to optimize the degrees of freedom parameter (ν) by comparing information criteria (AICc).

## Arguments

- `gas_model::GASModel`: The GAS model to be used for fitting the t-location-scale distribution.
- `y::Vector{Fl}`: Vector of observations.
- `X::Matrix{Fl}`: Matrix of additional regressors (optional).
- `tol::Float64 = 0.01`: Tolerance level for terminating the local search.
- `α::Float64 = 0.5`: Significance level for the robust fitting procedure.
- `robust_prop::Float64 = 0.7`: Proportion of data to be used in the robust fitting procedure.
- `number_max_iterations::Int64 = 30000`: Maximum number of iterations for the optimization procedure.
- `max_optimization_time::Float64 = 180.0`: Maximum time allowed for the optimization procedure.
- `initial_values::Union{Dict{String, Any}, Missing} = missing`: Initial values for optimization (optional).

## Returns

- best_model: The best-fitted model based on the t-location-scale distribution after the local search is determined by comparing information criteria (AICc).
"""
function fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl};
                                         tol::Float64 = 0.01, α::Float64 = 0.5, robust::Bool = false, robust_prop::Float64 = 0.7, 
                                         number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
                                         κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl}

    T    = length(y)
    dist = gas_model.dist

    fitted_model_ν, first_ν = find_first_model_for_local_search(gas_model, y, X;  α = α, robust = robust, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                                max_optimization_time =  max_optimization_time, initial_values = initial_values,
                                                                κ_min = κ_min, κ_max = κ_max)

    model_lower, parameters_lower, initial_values_lower = create_model(gas_model, y, X, first_ν-1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_lower = fit(gas_model, y, X, model_lower, parameters_lower, initial_values_lower; α = α, robust = robust, robust_prop = robust_prop)
    
    model_upper, parameters_upper, initial_values_upper = create_model(gas_model, y, X, first_ν+1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_upper = fit(gas_model, y, X, model_upper, parameters_upper, initial_values_upper; α = α, robust = robust, robust_prop = robust_prop)

    aicc_ = [is_valid_model(fitted_model_ν_lower) ? fitted_model_ν_lower.information_criteria["aicc"] : Inf,
             fitted_model_ν.information_criteria["aicc"],
             is_valid_model(fitted_model_ν_upper) ? fitted_model_ν_upper.information_criteria["aicc"] : Inf]

    aicc_[findall(i -> i == -Inf, aicc_)] .= Inf

    #historic_aicc = hcat([first_ν-1, first_ν, first_ν + 1],  aicc_)

    if argmin(aicc_) == 2
        @info("It was not necessary to do the local search!")
        #push!(historic_aicc, aicc_[2])
        best_model = fitted_model_ν
    else
        @info("Starting the local search!")
        search = true
        if argmin(aicc_) == 1
            factor     = -1
            current_ν  = first_ν - 1
            best_model = fitted_model_ν_lower
            best_aicc  = aicc_[1]

            @info("Current ν : $current_ν")
            @info("Best AICc: $best_aicc")
        else    
            factor     = 1
            current_ν  = first_ν + 1
            best_model = fitted_model_ν_upper
            best_aicc  = aicc_[3]

            @info("Current ν : $current_ν")
            @info("Best AICc: $best_aicc")
        end

        #push!(historic_aicc, best_aicc)

        while search && current_ν ≥ 3
            current_ν += 1 * factor
            @info("Trying ν = $current_ν")
            initial_values = create_output_initialization_from_fit(best_model, gas_model)

            model, parameter, _ = create_model(gas_model, y, X, current_ν;  number_max_iterations = number_max_iterations,
                                              max_optimization_time =  max_optimization_time, initial_values = initial_values)

            fitted_model   = fit(gas_model, y, X, model, parameter, initial_values; α = α, robust = robust, robust_prop = robust_prop)

            if (fitted_model.information_criteria["aicc"] - best_aicc) / best_aicc < -tol && is_valid_model(fitted_model)
                
                best_model = fitted_model
                best_aicc  = fitted_model.information_criteria["aicc"]
                #push!(historic_aicc, best_aicc)
                #vcat(historic_aicc, [current_ν  best_aicc])
                @info("Best AICc: $best_aicc")
            else 
                search = false
            end
            
        end
    end

    return best_model#, historic_aicc
end


