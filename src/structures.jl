abstract type ScoreDrivenDistribution end

"""
# GASModel

A mutable struct representing a Generalized Autoregressive Score (GAS) model.

## Fields
- `dist::ScoreDrivenDistribution`: The score-driven distribution used in the GAS model.
- `time_varying_params::Vector{Bool}`: A vector indicating which parameters are time-varying.
- `d::Union{Float64, Missing}`: The degree of freedom parameter. It can be a float or missing.
- `random_walk::Dict{Int64, Bool}`: A dictionary indicating whether random walk components are included for each parameter. It can be empty.
- `random_walk_slope::Dict{Int64, Bool}`: A dictionary indicating whether random walk slope components are included for each parameter. It can be empty.
- `ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}`: A dictionary indicating the autoregressive (AR) components included for each parameter. It can be empty.
- `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: A dictionary indicating whether seasonality components are included for each parameter. It can be empty.
- `robust::Bool`: A boolean indicating whether the model is robust.
- `stochastic::Bool`: A boolean indicating whether the model is stochastic.

## Constructor
- `GASModel(dist::ScoreDrivenDistribution, time_varying_params::Vector{Bool}, d::Union{Float64, Missing}, random_walk::Dict{Int64, Bool}, random_walk_slope::Dict{Int64, Bool}, ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, robust::Bool, stochastic::Bool)`: Constructs a new `GASModel` object with the specified parameters.

## Description
This struct represents a GAS model, which is a statistical model used for time series forecasting. It contains information about the distribution used, whether parameters are time-varying, the presence of various components like random walk, random walk slope, autoregressive (AR) components, and seasonality. Additionally, it indicates whether the model is robust and stochastic.

The dictionaries `random_walk`, `random_walk_slope`, `ar`, and `seasonality` can be empty, which indicates that the corresponding components are not included in the model.
"""
mutable struct GASModel
    dist::ScoreDrivenDistribution
    time_varying_params::Vector{Bool}
    d::Union{Float64, Missing}
    random_walk::Dict{Int64, Bool} 
    random_walk_slope::Dict{Int64, Bool} 
    ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}
    seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}
    robust::Bool
    stochastic::Bool

    function GASModel(dist::ScoreDrivenDistribution,
                        time_varying_params::Vector{Bool},
                        d::Union{Float64, Missing},
                        random_walk::Dict{Int64, Bool}, 
                        random_walk_slope::Dict{Int64, Bool}, 
                        ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}},
                        seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}},
                        robust::Bool, stochastic::Bool)

        num_params = length(time_varying_params)

        if length(random_walk) != num_params
            for i in 1:num_params
                if !haskey(random_walk, i)
                    random_walk[i] = false
                end
            end
        end

        if length(random_walk_slope) != num_params
            for i in 1:num_params
                if !haskey(random_walk_slope, i)
                    random_walk_slope[i] = false
                end
            end
        end

        if length(ar) != num_params
            for i in 1:num_params
                if !haskey(ar, i)
                    ar[i] = false
                end
            end
        end

        if length(seasonality) != num_params
            for i in 1:num_params
                if !haskey(seasonality, i)
                    seasonality[i] = false
                end
            end
        end

        return new(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)   
    end
end

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
