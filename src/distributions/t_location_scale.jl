"
Defines a Normal distribution with mean μ and variance σ².
"
mutable struct tLocationScaleDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
    ν::Union{Missing, Int64}
end

"
Outer constructor for the t location scale distribution.
"
function tLocationScaleDistribution()
    return tLocationScaleDistribution(missing, missing, missing)
end

"
Evaluate the score of a t location scale distribution with mean μ, scale parameter σ² and ν degrees of freedom, in observation y.
"
function score_tlocationscale(μ, σ², ν,  y) 
  
    return [((ν + 1) * (y - μ)) / ((y - μ)^2 + σ² * ν), -(ν * (σ² - (y - μ)^2)) / (2 * σ² * (ν * σ² + (y - μ)^2))]
end

"
Evaluate the fisher information of a t location scale distribution with mean μ, scale parameter σ² and ν degrees of freedom.
"
function fisher_information_tlocationscale(μ, σ², ν)
    return [((ν + 1.0)/(σ² * (ν + 3.0))) 0.0 ; 0.0 (ν / ((2 * σ²^2) * (ν + 3.0)))]
end

"
Evaluate the log pdf of a t location scale distribution with mean μ, scale parameter σ² and ν degrees of freedom, in observation y.
"
function logpdf_tlocationscale(μ, σ², ν, y)

    return logpdf_tlocationscale([μ, σ², ν], y)
end

"
Evaluate the log pdf of a t location scale distribution with mean μ, scale parameter σ² and ν degrees of freedom, in observation y.
    param[1] = μ
    param[2] = σ²
    param[3] = ν
"
function logpdf_tlocationscale(param, y)

    if param[2] < 0
        param[2] = 1e-4
    end

    if param[3] < 0
        param[3] = 3
    end

    return log(gamma((param[3] + 1)/2)) - log(gamma(param[3]/2)) - (1/2)*log(π*param[3]*param[2]) - (param[3] + 1)/2 * log(1 + ((y - param[1])^2)/(param[3]*param[2]))
end

"Evaluate the CDF t location scale distribution with mean μ, scale parameter σ² and ν degrees of freedom, in observation y"
function cdf_tlocationscale(param::Vector{Float64}, y::Fl) where Fl

    return Distributions.cdf(TDist(param[3]), (y - param[1]) / sqrt(param[2]))
end

"
Returns the code of the t location scale distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::tLocationScaleDistribution)
    return 2
end

"
Returns the number of parameters of the t location scale distribution.
"
function get_num_params(dist::tLocationScaleDistribution)
    return 3
end

"
Simulates a value from a given t location scale distribution.
    param[1] = μ
    param[2] = σ²
    param[3] = ν 
"
function sample_dist(param::Vector{Fl}, dist::tLocationScaleDistribution) where Fl
    
    if param[2] < 0
        param[2] = 1e-4
    end

    return param[1] + sqrt(param[2]) * rand(TDist(param[3]), 1)[1]
end

"
Fits a t-Student distribution and returns the degrees of freedom value.
"
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

    return Int64(round(value(ν))), value(σ²)
end

"
Indicates which parameters of the t location scale distribution must be positive.
"
function check_positive_constrainst(dist::tLocationScaleDistribution)
    return [false, true, true]
end

"
Returns a dictionary with the initial values of the parameters of the t location scale distribution that will be used in the model initialization.
"
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::tLocationScaleDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl

    T         = length(y)
    dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y ,ones(T) * var(diff(y)), (y.^2) ./ (ones(T) * var(diff(y))) , y, 0.5, dist_code, 2)).^2
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
 

"
Compare the two forms of initialize the ν parameter and return the fitted model and the ν with best performace in terms of AICC
"
function find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}; dates::Union{Nothing, Vector{Dl}} = nothing,
                                          α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    T    = length(y)
    dist = gas_model.dist

    optimal_ν, _ = find_ν(y, dist)
    heuristic_ν  = T - 1

    opt_model, opt_parameters, initial_values = create_model(gas_model, y, optimal_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values)

    optimal_model = fit(gas_model, y, opt_model, opt_parameters, initial_values; α = α, robust_prop = robust_prop)

    heu_model, heu_parameters, initial_values = create_model(gas_model, y, heuristic_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values)

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

"
Compare the two forms of initialize the ν parameter and return the fitted model and the ν with best performace in terms of AICC, with exogenous variables
"
function find_first_model_for_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; dates::Union{Nothing, Vector{Dl}} = nothing, 
                                          α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                          number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, 
                                          initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}
    T    = length(y)
    dist = gas_model.dist

    optimal_ν, _ = find_ν(y, dist)
    heuristic_ν  = T - 1

    opt_model, opt_parameters, initial_values = create_model(gas_model, y, X, optimal_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values)

    optimal_model = fit(gas_model, y, X, opt_model, opt_parameters, initial_values; α = α, robust_prop = robust_prop)

    heu_model, heu_parameters, initial_values = create_model(gas_model, y, X, heuristic_ν;  number_max_iterations = number_max_iterations,
                                             max_optimization_time =  max_optimization_time, initial_values = initial_values)

    heuristic_model = fit(gas_model, y, X, heu_model, heu_parameters, initial_values; α = α, robust_prop = robust_prop)

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

"
Use a local search method to find the best value of ν considering the AICC metric. 
"
function fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl}; dates::Union{Nothing, Vector{Dl}} = nothing,
                                            tol::Float64 = 0.01, α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
                                            number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    T    = length(y)
    dist = gas_model.dist

    fitted_model_ν, first_ν = find_first_model_for_local_search(gas_model, y;  α = α, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                                max_optimization_time =  max_optimization_time, initial_values = initial_values)

    model_lower, parameters_lower, initial_values_lower = create_model(gas_model, y,  first_ν-1; number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_lower = fit(gas_model, y, model_lower, parameters_lower, initial_values_lower; α = α, robust_prop = robust_prop)
    
    model_upper, parameters_upper, initial_values_upper = create_model(gas_model, y, first_ν+1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_upper = fit(gas_model, y, model_upper, parameters_upper, initial_values_upper; α = α, robust_prop = robust_prop)

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

            fitted_model   = fit(gas_model, y, model, parameter, initial_values; α = α, robust_prop = robust_prop)

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

"
Use a local search method to find the best value of ν considering the AICC metric, with exogenous variables. 
"
function fit_tlocationscale_local_search(gas_model::GASModel, y::Vector{Fl}, X::Matrix{Fl}; dates::Union{Nothing, Vector{Dl}} = nothing,
    tol::Float64 = 0.01, α::Float64 = 0.5, robust_prop::Float64 = 0.7, 
    number_max_iterations::Int64 = 30000, max_optimization_time::Float64 = 180.0, initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl, Dl}

    T    = length(y)
    dist = gas_model.dist

    fitted_model_ν, first_ν = find_first_model_for_local_search(gas_model, y, X;  α = α, robust_prop = robust_prop, number_max_iterations = number_max_iterations,
                                                                max_optimization_time =  max_optimization_time, initial_values = initial_values)

    model_lower, parameters_lower, initial_values_lower = create_model(gas_model, y, X, first_ν-1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_lower = fit(gas_model, y, X, model_lower, parameters_lower, initial_values_lower; α = α, robust_prop = robust_prop)
    
    model_upper, parameters_upper, initial_values_upper = create_model(gas_model, y, X, first_ν+1;  number_max_iterations = number_max_iterations,
                                                 max_optimization_time =  max_optimization_time, initial_values = initial_values)

    fitted_model_ν_upper = fit(gas_model, y, X, model_upper, parameters_upper, initial_values_upper; α = α, robust_prop = robust_prop)

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

            fitted_model   = fit(gas_model, y, X, model, parameter, initial_values; α = α, robust_prop = robust_prop)

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


