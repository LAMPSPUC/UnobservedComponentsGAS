abstract type ScoreDrivenDistribution end

"""
# GASModel

A mutable struct representing a Generalized Autoregressive Score (GAS) model.

## Fields
- `dist::ScoreDrivenDistribution`: The score-driven distribution used in the GAS model.
- `time_varying_params::Vector{Bool}`: A vector indicating which parameters are time-varying.
- `d::Union{Int64, Float64, Missing}`: The degree of freedom parameter. It can be an integer, float, or missing.
- `level::Union{String, Vector{String}}`: A string or a vector of strings indicating the level dynamics for each parameter. It can include "random walk", "random walk slope", "ar(1)", or be empty.
- `seasonality::Union{String, Vector{String}}`: A string or a vector of strings indicating the seasonality dynamics for each parameter. It includes the type and seasonal period, such as "deterministic 12", or can be empty.
- `ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Missing, Vector{Int64}}}}`: A vector indicating the autoregressive (AR) components included for each parameter. It can contain integers, missing values, or vectors of integers, or be missing.

## Constructor
- `GASModel(dist::ScoreDrivenDistribution, time_varying_params::Vector{Bool}, d::Union{Int64, Float64, Missing}, level::Union{String, Vector{String}}, seasonality::Union{String, Vector{String}}, ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Missing, Vector{Int64}}}})`: Constructs a new `GASModel` object with the specified parameters.

## Description
This struct represents a GAS model, which is a statistical model used for time series forecasting. It contains information about the distribution used, whether parameters are time-varying, the scale parameter, the level dynamics, seasonality dynamics, and autoregressive components for each parameter.

The `level` field indicates the dynamics of the level component, such as random walk, random walk slope, or AR(1), for each parameter. The `seasonality` field includes the type and seasonal period, such as "deterministic 12", for each parameter. The `ar` field indicates the autoregressive components for each parameter.

The constructor `GASModel` initializes a new GAS model with the provided parameters, ensuring the validity of the input values.
"""
mutable struct GASModel
    dist::ScoreDrivenDistribution
    time_varying_params::Vector{Bool}
    d::Union{Int64, Float64, Missing}
    level::Union{String, Vector{String}}
    seasonality::Union{String, Vector{String}} # includes type and seasonal period: "deterministic 12"
    ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}

    function GASModel(dist::ScoreDrivenDistribution,
        time_varying_params::Vector{Bool},
        d::Union{Int64, Float64, Missing},
        level::Union{String, Vector{String}},
        seasonality::Union{String, Vector{String}},# includes type and seasonal period: "deterministic 12"
        ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}},  Vector{Union{Missing, Vector{Int64}}}})

        num_params = length(time_varying_params)
        
        @assert d ∈ [1.0, 0.5, 0.0, 1, 0] "Invalid d value! It must be 1.0, 0.5 or 0.0"

        if typeof(level) == String
            level = vcat(level, fill("", num_params - 1))
        elseif typeof(level) == Vector{String}
            level = vcat(level, fill("", num_params - length(level)))
        else
            level = fill("", num_params)
        end
        
        if typeof(seasonality) == String
            seasonality = vcat(seasonality, fill("", num_params - 1))
        elseif typeof(seasonality) == Vector{String}
            seasonality = vcat(seasonality, fill("", num_params - length(seasonality)))
        else
            seasonality = fill("", num_params)
        end

        if ismissing(ar)
            ar = fill(missing, num_params)
        elseif length(ar) != num_params
            ar = vcat(ar, fill(missing, num_params - length(ar)))
        end

        @assert all([level[i] ∈ ["random walk", "random walk slope", "ar(1)", ""] for i in 1:num_params]) "Invalid level dynamic!"
        @assert all([split(seasonality[i], " ")[1] ∈ ["deterministic", "stochastic", ""] for i in 1:num_params]) "Invalid seasonality dynamic!"

        idx_fixed_params = findall(x -> x == false, time_varying_params)
        
        if all(.!isempty.(level[idx_fixed_params])) & !isempty(idx_fixed_params)
           @warn "Non missing level dynamic found for a fixed paramater. This dynamic will be ignored."
           level[time_varying_params .== false] .= ""
        end

        if all(.!isempty.(seasonality[idx_fixed_params])) & !isempty(idx_fixed_params)      
            @warn "Non missing seasonality dynamic found for a fixed paramater. This dynamic will be ignored."
            seasonality[time_varying_params .== false] .= ""
        end

        if all(.!ismissing.(ar[idx_fixed_params])) & !isempty(idx_fixed_params)
            @warn "Non missing ar dynamic found for a fixed paramater. This dynamic will be ignored."
            ar = Vector{Union{Missing, Int64}}(ar)
            ar[time_varying_params .== false] .= missing
        end

        return new(dist, time_varying_params, d, level, seasonality, ar)
        
    end
end

"""
A mutable struct representing the output of a GAS model estimation.

    ## Fields
    - `fit_in_sample::Vector{Float64}`: Vector of in-sample fits.
    - `fitted_params::Dict{String, Vector{Float64}}`: Dictionary containing the fitted parameters of the model.
    - `components::Dict{String, Any}`: Dictionary containing the components of the model.
    - `selected_variables::Union{Missing, Vector{Int64}}`: Selected variables for the model, or missing if not applicable.
    - `residuals::Dict{String, Any}`: Dictionary containing the residuals of the model.
    - `information_criteria::Dict{String, Float64}`: Dictionary containing information criteria values.
    - `penalty_factor::Float64`: Penalty factor used in model estimation.
    - `model_status::String`: Status of the model.
    
    ## Constructor
    - `Output(fit_in_sample::Vector{Float64}, fitted_params::Dict{String, Vector{Float64}}, components::Dict{String, Any}, selected_variables::Union{Missing, Vector{Int64}}, residuals::Dict{String, Any}, information_criteria::Dict{String, Float64}, penalty_factor::Float64, model_status::String)`: Constructs a new `Output` object with the specified parameters.
    
    ## Description
    This struct represents the output of a GAS model estimation, providing information about the fitted model, including in-sample fits, fitted parameters, model components, residuals, information criteria values, penalty factor, and model status.
"""
mutable struct Output
    fit_in_sample::Vector{Float64}
    fitted_params::Dict{String, Vector{Float64}}
    components::Dict{String, Any}
    selected_variables::Union{Missing, Vector{Int64}}
    residuals::Dict{String, Any}
    information_criteria::Dict{String, Float64}
    penalty_factor::Float64
    model_status::String
end 
