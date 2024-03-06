"
Defines the specified GAS model as a optimization problem
"
function create_model(gas_model::GASModel, y::Vector{Fl}, fixed_ν::Union{Missing, Int64};
    number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

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
    include_components!(model, s, gas_model, T);

    @info("Computing initial values...")
    if ismissing(initial_values)
        Random.seed!(123)
        initial_values = create_output_initialization(y, missing, gas_model);
    end

    @info("Including dynamics..")
    include_dynamics!(model,parameters, gas_model,  missing, T);

    if get_num_params(gas_model.dist) == 3
        register(model, :log_pdf, 4, DICT_LOGPDF[dist_name]; autodiff = true)
    elseif get_num_params(gas_model.dist) == 2
        register(model, :log_pdf, 3, DICT_LOGPDF[dist_name]; autodiff = true)
    end

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end

    return model, parameters, initial_values
end

"
Defines the specified GAS model with exogenous variables as a optimization problem
"
function create_model(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, fixed_ν::Union{Missing, Int64};
    number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

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
    s = compute_score(model, parameters,  y, d, time_varying_params, T, dist);
    
    @info("Including components...")
    include_components!(model, s, gas_model, T);

    @info("Computing initial values...")
    if ismissing(initial_values)
        Random.seed!(123)
        initial_values = create_output_initialization(y, X, gas_model)
    end

    @info("Including explanatory variables...")
    include_explanatory_variables!(model, X)

    @info("Including dynamics..")
    include_dynamics!(model, parameters, gas_model,  X, T)

    if get_num_params(gas_model.dist) == 3
        register(model, :log_pdf, 4, DICT_LOGPDF[dist_name]; autodiff = true)
    elseif get_num_params(gas_model.dist) == 2
        register(model, :log_pdf, 3, DICT_LOGPDF[dist_name]; autodiff = true)
    end

    # if log_normal_flag
    #     gas_model.dist = LogNormalDistribution()
    # end

    return model, parameters, initial_values
end

"
Evaluate if the model is based on t distribution or not and call the correct function to fit the model
"
function fit(gas_model::GASModel, y::Vector{Fl}; 
                α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing, tol::Float64 = 0.005) where Fl

    dist = gas_model.dist

    if typeof(dist) == tLocationScaleDistribution

        fitted_model = fit_tlocationscale_local_search(gas_model, y; 
                                                       α = α, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                       max_optimization_time = max_optimization_time, initial_values = initial_values)
    else
    
        model, parameters, initial_values = create_model(gas_model, y,missing;  number_max_iterations = number_max_iterations,
                                         max_optimization_time = max_optimization_time, initial_values = initial_values, tol = tol)

        fitted_model = fit(gas_model, y, model, parameters, initial_values; α = α, robust_prop = robust_prop)
    end

    return fitted_model
end

"
Evaluate if the model is based on t distribution or not and call the correct function to fit the model, with exogenous variables
"
function fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; 
                α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing,tol::Float64 = 0.005) where Fl

    dist = gas_model.dist

    if typeof(dist) == tLocationScaleDistribution
  
        fitted_model = fit_tlocationscale_local_search(gas_model, y, X;  
                                                       α = α, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                       max_optimization_time = max_optimization_time, initial_values = initial_values)
    else
        model, parameters, initial_values = create_model(gas_model, y, X, missing;  number_max_iterations = number_max_iterations,
                                         max_optimization_time = max_optimization_time, initial_values = initial_values, tol = tol)

        fitted_model = fit(gas_model, y, X, model, parameters, initial_values; α = α, robust_prop = robust_prop)
    end

    return fitted_model

end

"
Fits the specified GAS model.
"
function fit(gas_model::GASModel, y::Vector{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.5, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

    if typeof(gas_model.dist) == LogNormalDistribution
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end 

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    dist_code = get_dist_code(dist)
    
    T = length(y)

    @info("Including objective funcion...")
    include_objective_function!(model, parameters, y, T, robust, dist_code; α = α, robust_prop = robust_prop);

    @info("Initializing variables...")
    initialize_components!(model, initial_values, gas_model);

    @info("Optimizing the model...")
    optimize!(model)
    @info termination_status(model)

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end
  
    return create_output_fit(model, parameters, y, missing, missing, gas_model, α)
end

"
Fits the specified GAS model, with exogenous variables.
"
function fit(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, model::Ml, parameters::Matrix{Gl}, initial_values::Dict{String, Any}; α::Float64 = 0.5, robust_prop::Float64 = 0.7) where{Fl, Ml, Gl}

    if typeof(gas_model.dist) == LogNormalDistribution
        #println("Log Normal distribution")
        gas_model.dist = NormalDistribution()
        y = log.(y)
        log_normal_flag = true
    else
        log_normal_flag = false
    end

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    dist_code = get_dist_code(dist)
    
    T = length(y)
    
    @info("Including objective funcion...")
    include_objective_function!(model, parameters, y, T, robust, dist_code; α = α, robust_prop = robust_prop)

    @info("Initializing variables...")
    initialize_components!(model, initial_values, gas_model)

    @info("Optimizing the model...")
    optimize!(model)
    @info termination_status(model)

    if log_normal_flag
        gas_model.dist = LogNormalDistribution()
    end

    return create_output_fit(model, parameters, y, X, missing, gas_model, α)
end

# "
# Runs the automatic GAS. Finds the best value of α for d = 0, then find the best d for that fixed value of α.
# "
# function auto_gas(gas_model::GASModel, y::Vector{Fl}, steps_ahead::Int64; d_values::Vector{Float64} = [0.0, 0.5, 1.0],
#     robust_prop::Float64 = 0.7, number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
#     initial_values::Union{Dict{String, Any}, Missing} = missing, num_scenarious::Int64 = 500, 
#     probabilistic_intervals::Vector{Float64} = [0.8, 0.95], validation_horizont::Int64 = 12) where Fl

#     if typeof(gas_model.dist) == tLocationScaleDistribution

#         new_gas_model = deepcopy(gas_model)
#         new_gas_model.dist = NormalDistribution()
#         new_gas_model.time_varying_params = new_gas_model.time_varying_params[1:2]
#     else
#         new_gas_model = deepcopy(gas_model)
#     end

#     @info("Finding optimal value of α")

#     y_train = y[1:end-validation_horizont]
#     y_val   = y[end-(validation_horizont-1):end]

#     metrics  = []
#     alpha    =[]
#     times = []
#     aicc_aux = []
#     aiccs = []

#     if !ismissing(initial_values)
#         new_initial_values = get_part_initial_values(initial_values, validation_horizont)
#     else
#         new_initial_values = deepcopy(initial_values)
#     end
    
#     for d in d_values
#         println("Testando d = $d")
 
#         alfa = []

#         new_gas_model.d = d

#         model, parameters, first_initial_values = create_model(new_gas_model, y_train, missing; number_max_iterations = number_max_iterations,  
#                                 max_optimization_time = max_optimization_time, initial_values = new_initial_values, tol = 0.005);
        
#         new_initial_values = first_initial_values

#         function get_metric(α::Float64)
            
#             α = round(α; digits = 3)
#             println("Testando α = $α")
#             push!(alfa, α)

#             output = fit(new_gas_model, y_train, model, parameters, new_initial_values; α = α, robust_prop = robust_prop)
#             # @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust = new_gas_model

#             # dist_code =  get_dist_code(dist)

#             # T = length(y_train)
           
            
#             # include_objective_function!(model, parameters, log.(y_train), T, robust, dist_code; α = α, robust_prop = robust_prop);
            

#             # initialize_components!(model, new_initial_values, new_gas_model);
#             # optimize!(model)

#             # println(new_gas_model.dist)

#             # output = create_output_fit(model, parameters, y_train, missing, missing, new_gas_model, α)

#             #println("Funçao obj: ",objective_value(model))
    
#             if is_valid_model(output) && typeof(new_gas_model.dist) != LogNormalDistribution
#                 new_initial_values = create_output_initialization_from_fit(output, new_gas_model)
#             else
#                new_initial_values = first_initial_values 
#             end

#             try
#                 forec = predict(new_gas_model, output, y_train, validation_horizont, 100; probabilistic_intervals = [0.95])
#                 println("RMSE: ", sqrt(mean((y_val .- forec["mean"]).^2)))
                
#                 push!(aicc_aux, output.information_criteria["aicc"])

#                 return sqrt(mean((y_val .- forec["mean"]).^2)) #output.information_criteria["aicc"] #sqrt(mean((y_val .- forec["mean"]).^2))
#             catch
#                 return Inf
#             end
#         end

#         Random.seed!(123)
#         time = @elapsed optimization_result = Optim.optimize(get_metric, 0.0, 1.0, GoldenSection(); time_limit = 0.1, r_tol = 0.01, abs_tol = 0.001, iterations = 50)
    
#         push!(alpha, optimization_result.minimizer[1])
#         push!(metrics, Optim.minimum(optimization_result))
#         push!(times, time)
#         push!(aiccs, aicc_aux)

#     end
#     println("d = 0.0 => $(times[1]/60) minutos")
#     println("d = 0.5 => $(times[2]/60) minutos")
#     println("d = 1.0 => $(times[3]/60) minutos")
#     println(metrics)
#     println(alpha)
#     sorted_idx = sortperm(metrics)
#     gas_model.d = d_values[sorted_idx[1]]
 
#     println("Fitando modelo com d = $(gas_model.d)!!")

#     best_output = fit(gas_model, y;  α = alpha[sorted_idx[1]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                      number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)
    
#     forec = predict(gas_model, best_output, y, steps_ahead, num_scenarious; probabilistic_intervals = probabilistic_intervals)

#     historic_ampl = maximum(y) - minimum(y)

#     if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#         gas_model.d = d_values[sorted_idx[2]]

#         println("Fitando modelo com d = $(gas_model.d)!!")
 
#         best_output = fit(gas_model, y;  α = alpha[sorted_idx[2]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                      number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)

#         forec = predict(gas_model, best_output, y, steps_ahead, num_scenarious; probabilistic_intervals = probabilistic_intervals)

#         if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#             gas_model.d = d_values[sorted_idx[3]]

#             println("Fitando modelo com d = $(gas_model.d)!!")
    
#             best_output = fit(gas_model, y; α = alpha[sorted_idx[3]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                         number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)

#             forec = predict(gas_model, best_output, y,steps_ahead, num_scenarious; probabilistic_intervals = probabilistic_intervals)

#             if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#                 gas_model.d = d_values[sorted_idx[1]]
    
#                 println("Fitando modelo com d = $(gas_model.d)!!")
#                 best_output = fit(gas_model, y;  α = alpha[sorted_idx[1]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                                 number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)
                
#                 forec = predict(gas_model, best_output, y, steps_ahead, num_scenarious; probabilistic_intervals = probabilistic_intervals)   
#             end          
#         end
#     end

#     return best_output, gas_model, forec#, tested_alfa, metrics#, graficos

# end


# function auto_gas(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}, X_forecast::Matrix{Fl}, steps_ahead::Int64;  
#     d_values::Vector{Float64} = [0.0, 0.5, 1.0],
#     robust_prop::Float64 = 0.7, number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0,
#     initial_values::Union{Dict{String, Any}, Missing} = missing, num_scenarious::Int64 = 500,
#     probabilistc_interals::Vector{Float64} = [0.8, 0.95], validation_horizont::Int64 = 12) where Fl

#     if typeof(gas_model.dist) == tLocationScaleDistribution
#         new_gas_model = deepcopy(gas_model)
#         new_gas_model.dist = NormalDistribution()
#         new_gas_model.time_varying_params = new_gas_model.time_varying_params[1:2]
#     else
#         new_gas_model = deepcopy(gas_model)
#     end

#     @info("Finding optimal value of α")

#     y_train = y[1:end-validation_horizont]
#     y_val   = y[end-(validation_horizont-1):end]

#     X_train = X[1:end-validation_horizont, :]
#     X_val   = X[end-(validation_horizont-1):end, :]

#     metrics  = []
#     alpha    = []
#     times = []

#     if !ismissing(initial_values)
#         new_initial_values = get_part_initial_values(initial_values)
#     else
#         new_initial_values = deepcopy(initial_values)
#     end

#     for d in d_values

#         new_gas_model.d = d

#         model, parameters, first_initial_values  = create_model(new_gas_model, y_train, X_train, missing;number_max_iterations = number_max_iterations,  
#                                                max_optimization_time = max_optimization_time, initial_values = new_initial_values, tol = 0.005);

#         new_initial_values = first_initial_values 

#         function get_metric(α::Float64)

#             @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust = new_gas_model

#             dist_code =  get_dist_code(dist)

#             T = length(y_train)

#             α = round(α; digits = 3)
#             println("Testando α = $α")
            
#             println(new_gas_model.dist)

#             output = fit(new_gas_model, y_train, X_train, model, parameters, new_initial_values; α = α, robust_prop = robust_prop)
            
#             # include_objective_function!(model, parameters, y_train, T, robust, dist_code; α = α, robust_prop = robust_prop);
#             # initialize_components!(model, new_initial_values, new_gas_model);
#             # optimize!(model)

#             # output = create_output_fit(model, parameters, y_train, X_train, missing, new_gas_model, α)
#             # println("Funçao obj: ",objective_value(model))
    
#             if is_valid_model(output) && typeof(new_gas_model.dist) != LogNormalDistribution
#                 new_initial_values = create_output_initialization_from_fit(output, new_gas_model)
#             else
#                new_initial_values = first_initial_values 
#             end

#             println(output.model_status)

#             try 
#                 forec = predict(new_gas_model, output, y_train, X_val, validation_horizont, 100; probabilistic_intervals = [0.95])
#                 println("RMSE: ", sqrt(mean((y_val .- forec["mean"]).^2)))

#                 return sqrt(mean((y_val .- forec["mean"]).^2))
#             catch
#                 return Inf
#             end
#         end

#         time = @elapsed optimization_result = Optim.optimize(get_metric, 0.0, 1.0, GoldenSection(); r_tol = 0.01, abs_tol = 0.001, iterations = 50)
    
#         push!(alpha, optimization_result.minimizer[1])
#         push!(metrics, Optim.minimum(optimization_result))
#         push!(times, time)

#     end

#     println("d = 0.0 => $(times[1]/60) minutos")
#     println("d = 0.5 => $(times[2]/60) minutos")
#     println("d = 1.0 => $(times[3]/60) minutos")
#     println(metrics)
#     println(alpha)
#     sorted_idx = sortperm(metrics)
#     gas_model.d = d_values[sorted_idx[1]]
 
#     println("Fitando modelo com d = $(gas_model.d)!!")
#     best_output = fit(gas_model, y, X; α = alpha[sorted_idx[1]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                      number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)
    
#     forec = predict(gas_model, best_output, y, X_forecast, steps_ahead, num_scenarious; probabilistic_intervals = probabilistc_interals)
   
#     historic_ampl = maximum(y) - minimum(y)
   
#     if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#         gas_model.d = d_values[sorted_idx[2]]

#         println("Fitando modelo com d = $(gas_model.d)!!")
 
#         best_output = fit(gas_model, y, X; α = alpha[sorted_idx[2]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                      number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)

#         forec = predict(gas_model, best_output, y, X_forecast, steps_ahead, num_scenarious; probabilistic_intervals = probabilistc_interals)

    
#         if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#             gas_model.d = d_values[sorted_idx[3]]
#             println("Fitando modelo com d = $(gas_model.d)!!")
    
#             best_output = fit(gas_model, y, X;  α = alpha[sorted_idx[3]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                         number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)

#             forec = predict(gas_model, best_output, y, X_forecast, steps_ahead, num_scenarious; probabilistic_intervals = probabilistc_interals)
                       
#             if abs(minimum(forec["intervals"]["95"]["lower"]) - maximum(forec["mean"])) > 1.1*historic_ampl || abs(maximum(forec["intervals"]["95"]["upper"]) - minimum(forec["mean"])) > 1.1*historic_ampl
#                 gas_model.d = d_values[sorted_idx[1]]
    
#                 println("Fitando modelo com d = $(gas_model.d)!!")
#                 best_output = fit(gas_model, y, X;  α = alpha[sorted_idx[1]], robust_prop = robust_prop, max_optimization_time = max_optimization_time, 
#                                 number_max_iterations = number_max_iterations, initial_values = initial_values, tol = 0.000001)
                
#                 forec = predict(gas_model, best_output, y, X_forecast, steps_ahead, num_scenarious; probabilistic_intervals = probabilistc_interals)
#             end
#         end

#     end

#     return best_output, gas_model, forec
# end

"
Creates the output struct after fitiing the model.
"
function create_output_fit(model::Ml, parameters::Matrix{Gl} ,y::Vector{Fl}, X::Union{Missing, Matrix{Fl}}, selected_variables::Union{Missing, Vector{Int64}},  
                            gas_model::GASModel, penalty_factor::Float64) where {Ml, Gl, Fl}

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

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
        residuals = get_residuals(exp.(y), fit_in_sample, fitted_params, dist_code)
    else
        residuals = get_residuals(y, fit_in_sample, fitted_params, dist_code)
    end

    return Output(fit_in_sample, fitted_params, components, selected_variables, residuals, information_criteria, penalty_factor, String(Symbol(termination_status(model))))

end