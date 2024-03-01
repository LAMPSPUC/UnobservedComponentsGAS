"
Defines a Log Normal distribution with mean μ and variance σ².
"
mutable struct LogNormalDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
end

"
Outer constructor for the Normal distribution.
"
function LogNormalDistribution()
    return LogNormalDistribution(missing, missing)
end

"
Convert the fit in sample and the fitted params to the series scale.
"
function convert_to_exp_scale(fit_in_sample::Vector{Fl}, fitted_params::Dict{String, Vector{Float64}}) where Fl
     
    new_fit_in_sample = exp.(fit_in_sample .+ fitted_params["param_2"]./2)

    new_fitted_params = Dict{String, Vector{Float64}}()
    new_fitted_params["param_1"] = exp.(fitted_params["param_1"] .+ fitted_params["param_2"]./2)
    new_fitted_params["param_2"] = exp.(2*fitted_params["param_1"] .+ fitted_params["param_2"] ) .* (exp.(fitted_params["param_2"]) .- 1)

    return new_fit_in_sample, new_fitted_params
end

"
Convert the fitted params to the log scale.
"
function convert_to_log_scale(fitted_params::Dict{String, Vector{Float64}})

    new_fitted_params = Dict{String, Vector{Float64}}()
    new_fitted_params["param_1"] = log.(fitted_params["param_1"]) .- 0.5 .* log.(1 .+ fitted_params["param_2"]./(fitted_params["param_1"].^2))
    new_fitted_params["param_2"] = log.(1 .+ fitted_params["param_2"]./(fitted_params["param_1"].^2))

    return new_fitted_params
end

function convert_forecast_to_exp_scale(dict_forec::Dict{String, Any})

    dict_forec["mean"] = exp.( dict_forec["mean"])

    for s in 1:size(dict_forec["scenarios"], 2)
        dict_forec["scenarios"][:, s] = exp.(dict_forec["scenarios"][:, s])
    end

    for k in keys(dict_forec["intervals"])
        dict_forec["intervals"][k]["lower"] = exp.(dict_forec["intervals"][k]["lower"])
        dict_forec["intervals"][k]["upper"] = exp.(dict_forec["intervals"][k]["upper"])
    end
    
    return dict_forec
end

"
Returns the code of the Normal distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::LogNormalDistribution)
    return 1
end

"
Returns the number of parameters of the Normal distribution.
"
function get_num_params(dist::LogNormalDistribution)
    return 2
end

"
Simulates a value from a given Normal distribution.
    param[1] = μ
    param[2] = σ² 
"
function sample_dist(param::Vector{Float64}, dist::LogNormalDistribution)

    if param[2] < 0.0
        param[2] = 1e-4
    end

    return rand(Normal(param[1], sqrt(param[2])))
end
