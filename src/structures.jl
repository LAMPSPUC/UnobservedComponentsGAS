abstract type ScoreDrivenDistribution end

"
Specify the structure of the GAS model.
    dist: Conditional distribution to be considered
    time_varying_params: Vector of boolean that specify if the parameters is time-varying of not.
    d: Equal to 0.0, 0.5 or 1.0.
    random_walk: Dictionary that tells, for each time-varying parameter, if a random walk dynamic should be considered.
    random_walk_slope: Dictionary that tells, for each time-varying parameter, if a random walk dynamic with slope should be considered.
    ar: Dictionary that tells, for each time-varying parameter, if an autoregressive dynamic should be considered, with which lags.
    seasonality: Dictionary the tells, for each time-varying parameter, if seasonality should be considered, with which frequency.

    PS: The dictionaries can be empty.
"
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
