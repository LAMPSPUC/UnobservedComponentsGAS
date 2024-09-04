
"""
mutable struct LogNormalDistribution

    A mutable struct for representing the Log Normal distribution.

    # Fields
    - `μ::Union{Missing, Float64}`: Mean parameter.
    - `σ²::Union{Missing, Float64}`: Variance parameter.
"""

mutable struct LogNormalDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
end


"""
LogNormalDistribution()

Outer constructor for the Log-Normal distribution, with no arguments specified.
    
    # Returns
    - The LogNormalDistribution struct with both fields set to missing.
"""
function LogNormalDistribution()
    return LogNormalDistribution(missing, missing)
end

"""
convert_to_exp_scale(fit_in_sample::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}) where Fl

Convert the fit-in-sample vector and the dictionary of fitted parameters, which were originally computed in the Normal distribution scale, to the LogNormal distribution scale.
    
    # Arguments

    - `fit_in_sample::Vector{Fl}`: Model's fit-in-sample.
    - `fitted_params::Dict{String, Vector{Float64}}`: Dictionary mapping the estimated parameters of the distribution to their respective indices, for example, "param_1".
    
    # Returns
    - new_fit_in_sample: Vector{Float64}: The vector of the fit-in-sample in the LogNormal distribution scale.
    - new_fitted_params: Dict{String, Vector{Float64}}: Dictionary containing the values of the parameters in the LogNormal distribution scale.
        - "param_1": Vector containing the mean parameter of the LogNormal distribution. 
        - "param_2": Vector containing the variance parameter of the LogNormal distribution.
"""
function convert_to_exp_scale(fit_in_sample::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}) where Fl
     
    # fitted_params["param_2"] = fitted_params["param_2"] ./10000
    new_fit_in_sample = exp.(fit_in_sample .+ fitted_params["param_2"]./2)

    new_fitted_params = Dict{String, Vector{Float64}}()
    new_fitted_params["param_1"] = exp.(fitted_params["param_1"] .+ fitted_params["param_2"]./2)
    new_fitted_params["param_2"] = exp.(2*fitted_params["param_1"] .+ fitted_params["param_2"] ) .* (exp.(fitted_params["param_2"]) .- 1)

    return new_fit_in_sample, new_fitted_params
end

"""
convert_to_log_scale(fitted_params::Dict{String, Vector{Float64}})

Convert the dictionary of fitted parameters, which were originally in the LogNormal distribution scale, to the Normal distribution scale.
    
    # Arguments

    - `fitted_params::Dict{String, Vector{Float64}}`: Dictionary mapping the estimated parameters of the distribution to their respective indices, for example, "param_1".
    
    # Returns
    - new_fitted_params: Dict{String, Vector{Float64}}: Dictionary containing the values of the parameters in the LogNormal distribution scale.
        - "param_1": Vector containing the mean parameter of the Normal distribution. 
        - "param_2": Vector containing the variance parameter of the Normal distribution.
"""
function convert_to_log_scale(fitted_params::Dict{String, Vector{Float64}})

    new_fitted_params = Dict{String, Vector{Float64}}()
    new_fitted_params["param_1"] = log.(fitted_params["param_1"]) .- 0.5 .* log.(1 .+ fitted_params["param_2"]./(fitted_params["param_1"].^2))
    new_fitted_params["param_2"] = log.(1 .+ fitted_params["param_2"]./(fitted_params["param_1"].^2))

    return new_fitted_params
end

"""
convert_forecast_to_exp_scale(dict_forec::Dict{String, Any})

Convert the dictionary of forecasts, originally in the Normal distribution scale, to the LogNormal distribution scale.
    
    # Arguments

    - `dict_forec::Dict{String, Any}`: Dictionary mapping the predicted values to their corresponding indices.
    
    # Returns
    - new_dict_forec: Dict{String, Any}: Dictionary containing the predicted values in the LogNormal distribution scale.
        - "intervals": Dictionary containing vectors representing the upper and lower bounds of predictive intervals at a specified confidence level.
        - "mean": Vector containing the average of the simulated predicted values.
        - "scenarios": Matrix containing all the simulated predicted values.
"""

function convert_forecast_scenarios_to_exp_scale(scenarios::Matrix{Fl}) where Fl

    new_scenarios = deepcopy(scenarios)

    for s in 1:size(new_scenarios, 2)
        new_scenarios[:, s] = exp.(new_scenarios[:, s])
    end

    # new_dict_forec["mean"] = exp.(dict_forec["mean"])

    # for k in keys(dict_forec["intervals"])
    #     new_dict_forec["intervals"][k]["lower"] = exp.(dict_forec["intervals"][k]["lower"])
    #     new_dict_forec["intervals"][k]["upper"] = exp.(dict_forec["intervals"][k]["upper"])
    # end
    
    return new_scenarios
end

"""
get_dist_code(dist::LogNormalDistribution)

 Provide the code representing the Normal distribution, as the lognormal model is estimated based on the Gaussian one.
    
    # Arguments

    - `dist::LogNormalDistribution`: The structure that represents the LogNormal distribution.
    
    # Returns
    - The distribution code corresponding to the Normal distribution. (In this case, it always returns 1.)
"""
function get_dist_code(dist::LogNormalDistribution)
    return 1
end

"""
get_num_params(dist::LogNormalDistribution)

Provide the number of parameters for a LogNormal distribution.
    
    # Arguments

    - `dist::LogNormalDistribution`: The structure that represents the LogNormal distribution.
    
    # Returns
    - Int64(2)
"""
function get_num_params(dist::LogNormalDistribution)
    return 2
end

"""
sample_dist(param::Vector{Float64}, dist::LogNormalDistribution)

 Sample a random realization from a Normal distribution with the specified parameters.
    
    # Arguments

    -``param::Vector{Float64}`: A vector containing the parameters of the Normal distribution associated with the LogNormal distribution of interest.
        - param[1] = μ
        - param[2] = σ²
    - `dist::LogNormalDistribution`: The structure that represents the LogNormal distribution.
    
    # Returns
    - A realization of N(param[1], √param[2])
"""
function sample_dist(param::Vector{Float64}, dist::LogNormalDistribution)

    if param[2] < 0.0
        param[2] = 1e-4
    end

    return rand(Normal(param[1], sqrt(param[2])))
end
