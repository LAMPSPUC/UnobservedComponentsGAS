module UnobservedComponentsGAS

    using JuMP
    using Ipopt
    using LinearAlgebra
    using Parameters
    using Statistics
    using Distributions
    using Random 
    using SpecialFunctions
    using StateSpaceModels
    # For residuals diagnostics
    using Plots
    using HypothesisTests
    using StatsPlots
    using StatsBase
    using ARCHModels
     
    # include("../NonParametricStructuralModels/src/NonParametricStructuralModels.jl")

    include("structures.jl")
    include("distributions/common.jl")
    include("distributions/normal.jl")
    include("distributions/t_location_scale.jl")
    include("distributions/log_normal.jl")
    include("initialization.jl")
    include("fit.jl")
    include("utils.jl")
    include("components_dynamics.jl")
    include("optimization.jl")
    include("forecast.jl")
    include("residuals_diagnostics.jl")

    const DICT_CODE = Dict(1 => "Normal",
                           2 => "tLocationScale" )

    const DICT_SCORE = Dict("Normal"         => score_normal,
                            "tLocationScale" => score_tlocationscale)

    const DICT_FISHER_INFORMATION = Dict("Normal"         => fisher_information_normal,
                                         "tLocationScale" => fisher_information_tlocationscale)

    const DICT_LOGPDF = Dict("Normal"         => logpdf_normal,
                             "tLocationScale" => logpdf_tlocationscale)

    const DICT_CDF = Dict("Normal"         => cdf_normal,
                          "tLocationScale" => cdf_tlocationscale)
end 
