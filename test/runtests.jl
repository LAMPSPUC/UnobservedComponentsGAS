# Import all necessary packages
using UnobservedComponentsGAS
using Test, Random, Statistics, JuMP, Ipopt, Plots
using JSON3, CSV, DataFrames, StateSpaceModels


# Include all test files
include("test_fit_forecast_normal.jl")
include("test_fit_forecast_gamma.jl")
include("test_fit_forecast_lognormal.jl")
include("test_fit_forecast_t.jl")
include("test_components_dynamics.jl")
include("test_optimization.jl")
include("test_distributions.jl")
include("test_initialization.jl")
include("test_residuals_diagnostics.jl")