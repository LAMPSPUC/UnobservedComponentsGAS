using CSV, DataFrames, Plots, Distributions
using Metrics, Random
using Base.Threads
import Pkg
Pkg.activate(".")
Pkg.instantiate()
include("src/UnobservedComponentsGAS.jl")

function read_dataframes(granularity::String)::Tuple{DataFrame, DataFrame}

    train_set = CSV.read("$granularity-train.csv", DataFrame)
    test_set  =  CSV.read("$granularity-test.csv", DataFrame)

    return train_set, test_set
end

function read_MEB()::Tuple{DataFrame, DataFrame}

    train_set =  CSV.read("SimulacaoMEB_Train.csv", DataFrame)
    test_set  =  CSV.read("SimulacaoMEB_Test.csv", DataFrame)

    return train_set, test_set
end

function build_train_test_dict(df_train::DataFrame, df_test::DataFrame, N::Union{Int64,Missing})::Dict{Int, Dict{String, Vector{Float64}}}
    train_test_dict = Dict()
    # Adicionar codigo para selecionar apenas uma amostra das series
    
    Random.seed!(0)
    ismissing(N) ? idx = collect(1:size(df_train,1)) : idx = rand(1:size(df_train,1),N)

    println(size(df_train))
    println(size(df_test))
    df_train = df_train[idx,:]
    df_test = df_test[idx,:]
    println(size(df_train))
    println(size(df_test))

    for i in eachindex(df_train[:, 1])
        y_raw = Vector(df_train[i, :])#[2:end]
        y_train_raw = y_raw[1:findlast(i->!ismissing(i), y_raw)]
        T = length(y_train_raw)
        y_train = y_train_raw
        y_test  = Vector(df_test[i, :])#[2:end]

        train_test_dict[i] = Dict()
        train_test_dict[i]["train"] = Float64.(y_train)
        train_test_dict[i]["test"]  = Float64.(y_test)
    end

    return train_test_dict
end


function MASE(y_train::Vector{Fl}, y_test::Vector{Fl}, y_forecast::Vector{Fl}; s::Int64=12)::Float64 where {Fl}
    T = length(y_train)
    H = length(y_test)

    numerator   = (1/H) * sum(abs(y_test[i] - y_forecast[i]) for i in 1:H)
    denominator = (1/(T - s)) * sum(abs(y_train[j] - y_train[j - s]) for j in s+1:T)
    return numerator/denominator
end

granularity = "Monthly"
N = 100
# data_dict  = build_train_test_dict(read_dataframes(granularity)..., N);

data_dict  = build_train_test_dict(read_MEB()..., N);

timeout = 300
deterministic = true
deterministic ? seasonality = "deterministic 12" : seasonality = "stochastic 12"
d = 1.0
d2 = "d1"
folder = "results_model3_sm/"
model = "model3_sm"

df_results = DataFrame([[],[],[], [], [], [], [], [], []],
                     ["serie", "T", "model", "t create", "t optim", "rmse train", "rmse test", "mase test", "status"])

for n in 1:N
    println("   Série = $n")
    y_train = data_dict[n]["train"]
    y_test  = data_dict[n]["test"]
    T       = length(y_train)
    plot(vcat(y_train, y_test))

    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], d, "random walk slope", seasonality, 1)
    t_optimp = @elapsed  fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train; number_max_iterations = 50000, max_optimization_time = 300.0);
    # println(fitted_model.model_status)
    forecp = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, 18, 500);
    masep = MASE(y_train, y_test, forecp["mean"])
    rmse_paramp = sqrt(Metrics.mse(fitted_model.fit_in_sample, y_train))
    rmsep = sqrt(Metrics.mse(forecp["mean"], y_test))

    push!(df_results, [n, T, model, -1, t_optimp, rmse_paramp, rmsep, masep, fitted_model.model_status])

    CSV.write(folder*"results_$(model)_$(d2).csv", df_results)
end   


