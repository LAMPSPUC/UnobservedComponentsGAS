# Import all necessary packages
using UnobservedComponentsGAS
using Test, Random, Statistics, JuMP, Ipopt
using JSON3, CSV, DataFrames, StateSpaceModels


# Include all test files
include("test_fit.jl")
include("test_forecast.jl")
include("test_components_dynamics.jl")
include("test_optimization.jl")
include("test_distributions.jl")
include("test_initialization.jl")


