# Import all necessary packages
using UnobservedComponentsGAS
using Test, Random, Statistics, JuMP
using JSON3, CSV, DataFrames, StateSpaceModels


# Include all test files
include("test_update_model.jl")
include("test_forecast.jl")
include("test_distributions.jl")
include("test_initialization.jl")
include("test_fit.jl")