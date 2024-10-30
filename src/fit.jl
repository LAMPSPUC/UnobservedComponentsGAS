"""
# create_model(gas_model::GASModel, y::Vector{Fl}, fixed_ν::Union{Missing, Int64};
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
                initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

Creates a generalized autoregressive score (GAS) model based on the given model's specifications and data.

## Arguments
- `gas_model::GASModel`: The GAS model containing model's specifications.
- `y::Vector{Fl}`: The time series data to be modeled.
- `fixed_ν::Union{Missing, Int64}`: The fixed degrees of freedom parameter for the GAS model.
- `number_max_iterations::Int64`: The maximum number of iterations for optimization. Default is 30000.
- `max_optimization_time::Float64`: The maximum CPU time allowed for optimization. Default is 180.0 seconds.
- `initial_values::Union{Dict{String, Any}, Missing}`: Initial values for the model parameters. Default is `missing`.
- `tol::Float64`: Tolerance for optimization convergence. Default is 0.005.

## Returns
- `model`: The created JuMP model representing the GAS model.
- `parameters`: The parameters included in the model.
- `initial_values`: The initial values for the model parameters.
"""
function create_model(gas_model::GASModel, y::Vector{Fl}, fixed_ν::Union{Missing, Int64};
    number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
    κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, κ_max_s::Union{Float64, Int64} = 1,
    fix_num_harmonic::Vector{U} = [missing, missing], initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where {Fl, U}

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    # @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model
    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    dist_code = get_dist_code(dist)
    dist_name = DICT_CODE[dist_code]
    
    T = length(y)

    @info("Creating GAS model...")
    model = JuMP.Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", number_max_iterations)
    set_optimizer_attribute(model, "max_cpu_time", max_optimization_time)
    set_optimizer_attribute(model, "tol", tol)
    set_silent(model)

    @info("Including parameters...")
    parameters = include_parameters(model, time_varying_params, T, dist, fixed_ν);

    @info("Computing score...")
    s = compute_score(model, parameters, y, d, time_varying_params, T, dist);
    
    @info("Including components...")
    include_components!(model, s, gas_model, T; κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s, fix_num_harmonic = fix_num_harmonic);

    @info("Computing initial values...")
    if ismissing(initial_values)
        Random.seed!(123)
        initial_values = create_output_initialization(y, missing, gas_model; fix_num_harmonic = fix_num_harmonic);
    end

    @info("Including dynamics..")
    include_dynamics!(model,parameters, gas_model,  missing, T);

    # if get_num_params(gas_model.dist) == 3
    #     # @register(model, :log_pdf, 4, DICT_LOGPDF[dist_name]; autodiff = true)
    #     @operator(model, log_pdf, 4, DICT_LOGPDF[dist_name])
    # elseif get_num_params(gas_model.dist) == 2
    #     # @register(model, :log_pdf, 3, DICT_LOGPDF[dist_name]; autodiff = true)
    #     @operator(model, log_pdf, 3, DICT_LOGPDF[dist_name])
    # end

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end

    return model, parameters, initial_values
end

"""
# create_model(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, fixed_ν::Union{Missing, Int64};
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
                initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

Creates a generalized autoregressive score (GAS) model with explanatory variables based on the given model's specifications and data.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `y::Vector{Fl}`: The dependent variable time series data to be modeled.
- `X::Matrix{Fl}`: The matrix containing explanatory variables.
- `fixed_ν::Union{Missing, Int64}`: The fixed degrees of freedom parameter for the GAS model.
- `number_max_iterations::Int64`: The maximum number of iterations for optimization. Default is 30000.
- `max_optimization_time::Float64`: The maximum CPU time allowed for optimization. Default is 180.0 seconds.
- `initial_values::Union{Dict{String, Any}, Missing}`: Initial values for the model parameters. Default is `missing`.
- `tol::Float64`: Tolerance for optimization convergence. Default is 0.005.

## Returns
- `model`: The created JuMP model representing the GAS model with explanatory variables.
- `parameters`: The parameters included in the model.
- `initial_values`: The initial values for the model parameters.
"""
function create_model(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, fixed_ν::Union{Missing, Int64};
    number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
    κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, κ_max_s::Union{Float64, Int64} = 1,
    fix_num_harmonic::Vector{U} = [missing, missing], initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where {Fl, U}

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    # @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model
    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    dist_code = get_dist_code(dist)
    dist_name = DICT_CODE[dist_code]
    
    T = length(y)

    #@info("Creating GAS model...")
    model = JuMP.Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", number_max_iterations)
    set_optimizer_attribute(model, "max_cpu_time", max_optimization_time)
    set_optimizer_attribute(model, "tol", tol)
    set_silent(model)

    #@info("Including parameters...")
    parameters = include_parameters(model, time_varying_params, T, dist, fixed_ν);

    #@info("Computing score...")
    s = compute_score(model, parameters,  y, d, time_varying_params, T, dist);
    
    #@info("Including components...")
    include_components!(model, s, gas_model, T; κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s, fix_num_harmonic = fix_num_harmonic);

    #@info("Computing initial values...")
    if ismissing(initial_values)
        Random.seed!(123)
        initial_values = create_output_initialization(y, X, gas_model; fix_num_harmonic = fix_num_harmonic)
    end

    #@info("Including explanatory variables...")
    include_explanatory_variables!(model, X)

    #@info("Including dynamics..")
    include_dynamics!(model, parameters, gas_model,  X, T)

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end

    return model, parameters, initial_values
end

"""
## fit(gas_model::GASModel, y::Vector{Fl}; α::Float64 = 0.0, robust::Bool = false, robust_prop::Float64 = 0.7, number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

Fits the specified GAS (Generalized AutoRegressive Conditional Heteroskedasticity) model to the given time series data.

### Arguments
- `gas_model::GASModel`: The GAS model to fit to the data.
- `y::Vector{Fl}`: A vector representing the time series data.
- `α::Float64`: The significance level for the optimization process. Default is `0.5`.
- `robust::Bool`: Whether to use robust optimization. Default is `false`.
- `robust_prop::Float64`: Proportion of observations to be considered for robust optimization. Default is `0.7`.
- `number_max_iterations::Int64`: Maximum number of iterations for optimization. Default is `30000`.
- `max_optimization_time::Float64`: Maximum optimization time in seconds. Default is `180.0`.
- `initial_values::Union{Dict{String, Any}, Missing}`: Initial values for optimization. Default is `missing`.
- `tol::Float64`: Tolerance level for convergence. Default is `0.005`.

### Returns
- `fitted_model`: The fitted GAS model.

## Details
- If the distribution of the GAS model is `tLocationScaleDistribution`, it fits the model using local search to optimize the degrees of freedom parameter (ν).
- Otherwise, it creates a GAS model based on the specifications and fits it to the data.
"""
function fit(gas_model::GASModel, y::Vector{Fl}; 
                α::Float64 = 0.0, robust::Bool = false, robust_prop::Float64 = 0.7, 
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                fix_num_harmonic::Vector{U} = [missing, missing], κ_max_s::Union{Float64, Int64} = 1,
                κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where {Fl, U}

    dist = gas_model.dist
    
    if typeof(dist) == tLocationScaleDistribution

        fitted_model = fit_tlocationscale_local_search(gas_model, y; 
                                                       α = α, robust = robust ,robust_prop = robust_prop, 
                                                       number_max_iterations = number_max_iterations, κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s,
                                                       max_optimization_time = max_optimization_time, initial_values = initial_values, 
                                                       fix_num_harmonic = fix_num_harmonic)
    else
    
        model, parameters, initial_values = create_model(gas_model, y,missing;  number_max_iterations = number_max_iterations,
                                         max_optimization_time = max_optimization_time, initial_values = initial_values, tol = tol,
                                         κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s, fix_num_harmonic = fix_num_harmonic)

        fitted_model = fit(gas_model, y, model, parameters, initial_values; α = α, robust = robust, robust_prop = robust_prop)
    end

    return fitted_model
end

"""
## fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; α::Float64 = 0.0, robust::Bool=false, robust_prop::Float64 = 0.7, number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing,tol::Float64 = 0.005) where Fl

Fits the specified GAS (Generalized AutoRegressive Conditional Heteroskedasticity) model with exogenous variables to the given time series data.

### Arguments
- `gas_model::GASModel`: The GAS model to fit to the data.
- `y::Vector{Fl}`: A vector representing the time series data.
- `X::Matrix{Fl}`: A matrix representing the exogenous variables data.
- `α::Float64`: The significance level for the optimization process. Default is `0.5`.
- `robust::Bool`: Whether to use robust optimization. Default is `false`.
- `robust_prop::Float64`: Proportion of observations to be considered for robust optimization. Default is `0.7`.
- `number_max_iterations::Int64`: Maximum number of iterations for optimization. Default is `30000`.
- `max_optimization_time::Float64`: Maximum optimization time in seconds. Default is `180.0`.
- `initial_values::Union{Dict{String, Any}, Missing}`: Initial values for optimization. Default is `missing`.
- `tol::Float64`: Tolerance level for convergence. Default is `0.005`.

### Returns
- `fitted_model`: The fitted GAS model.

## Details
- If the distribution of the GAS model is `tLocationScaleDistribution`, it fits the model using local search to optimize the degrees of freedom parameter (ν).
- Otherwise, it creates a GAS model based on the specifications and fits it to the data.
"""
function fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; 
                α::Float64 = 0.0, robust::Bool=false, robust_prop::Float64 = 0.7, 
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2, κ_max_s::Union{Float64, Int64} = 1,
                fix_num_harmonic::Vector{U} = [missing, missing], initial_values::Union{Dict{String, Any}, Missing} = missing,tol::Float64 = 0.005) where {Fl, U}

    dist = gas_model.dist

    if typeof(dist) == tLocationScaleDistribution
  
        fitted_model = fit_tlocationscale_local_search(gas_model, y, X;  
                                                       tol = 0.01, α = α, robust = robust, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                       max_optimization_time = max_optimization_time, initial_values = initial_values,
                                                       κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s, fix_num_harmonic = fix_num_harmonic)
    else
        model, parameters, initial_values = create_model(gas_model, y, X, missing;  number_max_iterations = number_max_iterations,
                                         max_optimization_time = max_optimization_time,  tol = tol,
                                         κ_min = κ_min, κ_max = κ_max, κ_max_s = κ_max_s, initial_values = initial_values, fix_num_harmonic = fix_num_harmonic)

        fitted_model = fit(gas_model, y, X, model, parameters, initial_values; α = α, robust = robust, robust_prop = robust_prop)
    end

    return fitted_model

end

"""
## fit(gas_model::GASModel, y::Vector{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.0, robust::Bool=false, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

Fits the specified GAS (Generalized AutoRegressive Conditional Heteroskedasticity) model to the given time series data.

### Arguments
- `gas_model::GASModel`: The GAS model to fit to the data.
- `y::Vector{Fl}`: A vector representing the time series data.
- `model::Ml`: The optimization model.
- `parameters::Matrix{Gl}`: Matrix of parameters for optimization.
- `initial_values::Dict{String, Any}`: Initial values for optimization.
- `α::Float64`: The significance level for the optimization process. Default is `0.5`.
- `robust::Bool`: Whether to use robust optimization. Default is `false`.
- `robust_prop::Float64`: Proportion of observations to be considered for robust optimization. Default is `0.7`.

### Returns
- `fitted_model`: The fitted GAS model.

## Details
- If the distribution of the GAS model is `LogNormalDistribution`, it transforms the dependent variable data to natural logarithms.
- Includes the objective function, initializes variables, optimizes the model, and returns the fitted GAS model.
"""
function fit(gas_model::GASModel, y::Vector{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.0, robust::Bool=false, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    # @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model
    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    dist_code = get_dist_code(dist)
    
    T = length(y)

    @info("Including objective funcion...")
    include_objective_function!(model, parameters, y, T, robust, dist_code; α = α, robust_prop = robust_prop);

    @info("Initializing variables...")
    initialize_components!(model, initial_values, gas_model; );

    @info("Optimizing the model...")
    optimize!(model)
    @info termination_status(model)

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end
  
    return create_output_fit(model, parameters, y, missing, missing, gas_model, α)
end

"""
## fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.0, robust::Bool = false, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

Fits the specified GAS (Generalized AutoRegressive Conditional Heteroskedasticity) model to the given time series data with exogenous variables.

### Arguments
- `gas_model::GASModel`: The GAS model to fit to the data.
- `y::Vector{Fl}`: A vector representing the dependent variable time series data.
- `X::Matrix{Fl}`: A matrix representing the exogenous variables time series data.
- `model::Ml`: The optimization model.
- `parameters::Matrix{Gl}`: Matrix of parameters for optimization.
- `initial_values::Dict{String, Any}`: Initial values for optimization.
- `α::Float64`: The significance level for the optimization process. Default is `0.5`.
- `robust::Bool`: Whether to use robust optimization. Default is `false`.
- `robust_prop::Float64`: Proportion of observations to be considered for robust optimization. Default is `0.7`.

### Returns
- `fitted_model`: The fitted GAS model.

## Details
- If the distribution of the GAS model is `LogNormalDistribution`, it transforms the dependent variable data to natural logarithms.
- Includes the objective function, initializes variables, optimizes the model, and returns the fitted GAS model.
"""
function fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.0, robust::Bool = false, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    dist_code = get_dist_code(dist)
    
    T = length(y)
    
    #@info("Including objective funcion...")
    include_objective_function!(model, parameters, y, T, robust, dist_code; α = α, robust_prop = robust_prop)

    #@info("Initializing variables...")
    initialize_components!(model, initial_values, gas_model)

    #@info("Optimizing the model...")
    optimize!(model)
    #@info termination_status(model)

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end

    return create_output_fit(model, parameters, y, X, missing, gas_model, α)
end

"""
# create_output_fit(model::Ml, parameters::Matrix{Gl} ,y::Vector{Fl}, X::Union{Missing, Matrix{Fl}}, selected_variables::Union{Missing, Vector{Int64}},  
                            gas_model::GASModel, penalty_factor::Float64) where {Ml, Gl, Fl}

Creates an output structure containing the fitted values, residuals, and information criteria of a GAS model.

## Arguments
- `model::Ml`: The fitted optimization model.
- `parameters::Matrix{Gl}`: The optimized parameters of the GAS model.
- `y::Vector{Fl}`: The dependent variable time series data.
- `X::Union{Missing, Matrix{Fl}}`: The matrix of explanatory variables. Default is `missing`.
- `selected_variables::Union{Missing, Vector{Int64}}`: The indices of selected variables. Default is `missing`.
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `penalty_factor::Float64`: The penalty factor used for regularization.

## Returns
- `Output`: A structure containing the fitted values, residuals, and information criteria.
"""
function create_output_fit(model::Ml, parameters::Matrix{Gl} ,y::Vector{Fl}, X::Union{Missing, Matrix{Fl}}, selected_variables::Union{Missing, Vector{Int64}},  
                            gas_model::GASModel, penalty_factor::Float64) where {Ml, Gl, Fl}

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    dist_code  = get_dist_code(dist)
    num_params = get_num_params(dist)

    order  = []
    for i in 1:num_params
        if typeof(ar[i]) == Int64
            push!(order, collect(1:ar[i]))
        elseif typeof(ar[i]) == Vector{Int64}
            push!(order, ar[i])
        end
    end

    if isempty(order)
        first_idx = 2
    else 
        first_idx = maximum(vcat(order...)) + 1
    end

    information_criteria = get_information_criteria(model, parameters, y, dist)

    fit_in_sample, fitted_params, components = get_fitted_values(gas_model, model,  X)

    if typeof(dist) == LogNormalDistribution
        fit_in_sample, fitted_params = convert_to_exp_scale(fit_in_sample, fitted_params)
        residuals = get_residuals(exp.(y), fit_in_sample, fitted_params, dist)
    else
        residuals = get_residuals(y, fit_in_sample, fitted_params, dist)
    end

    return Output(fit_in_sample, fitted_params, components, selected_variables, residuals, information_criteria, penalty_factor, String(Symbol(termination_status(model))))

end