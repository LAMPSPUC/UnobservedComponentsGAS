using JuMP, Ipopt, CSV, DataFrames, Plots, Distributions
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

function logpdf_normal(μ, σ², y)
    return logpdf_normal([μ, σ²], y)
end

function logpdf_normal(param, y)

    if param[2] < 0
        param[2] = 1e-4
    end

    return -0.5 * log(2 * π * param[2]) - ((y - param[1])^2)/(2 * param[2])
end

function define_model_param_variable_dyn_expression(y, solver, deterministic, initial_values)
    T = length(y)
    model = JuMP.Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_cpu_time",300.0)
    set_optimizer_attribute(model, "max_iter", 50000)
    set_optimizer_attribute(model, "tol", 5e-3)

    #criando parametros da distribuição
    @variable(model, params[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RW1)
    @variable(model, -5 <= κ_RWS[1] <= 5)
    @variable(model, b1)
    @variable(model, -5 <= κ_b[1] <= 5)
    @variable(model, S1)

    set_start_value.(model[:params], y)
    #parameters = Matrix(undef, T, 2)
    #parameters[:, 1] .= model[:params]
    #parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    #s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution())
    @variable(model, s[1:T])
    # @constraint(model, [t = 1:T], s[t] == y[t] - model[:params][t]) # d = 1
    # @constraint(model, [t = 1:T], s[t] * sqrt(model[:fixed_params][2]) == y[t] - model[:params][t]) # d = 0.5
    # @constraint(model, [t = 1:T], s[t] * model[:fixed_params][2] == y[t] - model[:params][t]) # d = 0
    @operator(model, scaled_score_int, 6, UnobservedComponentsGAS.scaled_score)
    @constraint(model, [t = 1:T], s[t] == scaled_score_int(model[:params][t], model[:fixed_params][2],y[t], 0.5, 1, 1))

    # [(y - μ)/σ²; -(0.5/σ²) * (1 - ((y - μ)^2)/σ²)]

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
    else
        @variable(model, -5 <= κ_S[1] <= 5)
        @variable(model, γ_sto1[1:6, 1])
        @variable(model, γ_star_sto1[1:6, 1])
        γ_sto            = Matrix(undef, 6, T)
        γ_star_sto       = Matrix(undef, 6, T)
        γ_sto[:, 1]      = γ_sto1
        γ_star_sto[:, 1] = γ_star_sto1
    end

    RWS = Vector(undef, T)
    RWS[1] = model[:RW1]

    b = Vector(undef, T)
    b[1] = model[:b1]

    S = Vector(undef, T)
    S[1] = model[:S1]

    for t in 2:T
        RWS[t] = @expression(model, RWS[t-1] + b[t-1] + model[:κ_RWS][1]*model[:s][t])
        b[t]   = @expression(model, b[t-1] + model[:κ_b][1]*model[:s][t])
        if deterministic
            S[t] = @expression(model, sum(γ_det[i, 1]*cos(2 * π * i * t/12) + γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6))
        else
            for i in 1:6
                γ_sto[i, t] = @expression(model,  γ_sto[i, t-1] * cos(2*π*i / 12)  + γ_star_sto[i,t-1]*sin(2*π*i / 12) + κ_S[1] *model[:s][t])
                γ_star_sto[i, t] = @expression(model, -γ_sto[i, t-1] * sin(2*π*i / 12) + γ_star_sto[i,t-1]*cos(2*π*i / 12) + κ_S[1] *model[:s][t])
            end
            S[t] = @expression(model, sum(γ_sto[i, t]  for i in 1:6))
        end
        drop_zeros!(RWS[t])
        drop_zeros!(b[t])
        drop_zeros!(S[t])
    end

    @expression(model, RWS, RWS);
    @expression(model, b, b);
    @expression(model, S, S);
    @constraint(model, [t in 1:T], params[t] == RWS[t] + S[t]);

    set_start_value.(model[:params], round.(initial_values["param"]; digits = 5))
    set_start_value.(model[:fixed_params], round.(initial_values["fixed_param"]; digits = 5))
    set_start_value.(model[:s], round.(UnobservedComponentsGAS.scaled_score.(initial_values["param"][:, 1], initial_values["fixed_param"][1], y, 1.0, 1, 1); digits = 5))
   
    @operator(model, log_pdf, 3, logpdf_normal)
    #@variable(model, θ[1:T])
    #@constraint(model, [t=1:T], model[:θ][t] ≤  s[t])
    #@constraint(model, [t=1:T], model[:θ][t] ≤ -s[t])
    @objective(model, Max, sum(log_pdf(model[:params][t],model[:fixed_params][2], y[t]) for t in 1:T));
    return model
end

function define_model_param_expression_dyn_variable(y, solver, deterministic, initial_values, seasonality)
    T = length(y)
    model = JuMP.Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 50000)
    set_optimizer_attribute(model, "max_cpu_time", 300.)
    set_optimizer_attribute(model, "tol", 5e-3)
    #criando parametros da distribuição
    #@variable(model, μ[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RWS[1:T])
    @variable(model, -5 <= κ_RWS[1] <= 5)
    @variable(model, b[1:T])
    @variable(model, -5 <= κ_b[1] <= 5)
    @variable(model, S[1:T])

    # set_start_value.(model[:RWS], y)

    @expression(model, params, RWS + S)

    # parameters = Matrix(undef, T, 2)
    # parameters[:, 1] .= model[:params]
    # parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    #s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution());
    @variable(model, s[1:T])
    # @constraint(model, [t = 1:T], s[t] == y[t] - model[:params][t]) # d = 1
    # @constraint(model, [t = 1:T], s[t] * sqrt(model[:fixed_params][2]) == y[t] - model[:params][t]) # d = 0.5
    # @constraint(model, [t = 1:T], s[t] * model[:fixed_params][2] == y[t] - model[:params][t]) # d = 0
    @operator(model, scaled_score_int, 6, UnobservedComponentsGAS.scaled_score)
    @constraint(model, [t = 1:T], s[t] == scaled_score_int(model[:params][t], model[:fixed_params][2],y[t], 0.5, 1, 1))

    @constraint(model, [t = 2:T], b[t] == b[t - 1] + κ_b[1] * s[t]) #* s[1][t])
    @constraint(model, [t = 2:T], RWS[t] == RWS[t - 1] + b[t - 1] + κ_RWS[1]* s[t]) #* s[1][t])

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
        @constraint(model, [t = 1:T], S[t] == sum(γ_det[i, 1]*cos(2 * π * i * t/12) + 
                                                γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6))
    else
        @variable(model, κ_S[1])
        @constraint(model, [i in 1], -5 ≤ κ_S[1] ≤ 5)    

        @variable(model, γ_sto[1:6, 1:T, 1])
        @variable(model, γ_star_sto[1:6, 1:T, 1])

        @constraint(model, [i = 1:6, t = 2:T], γ_sto[i, t, 1] == γ_sto[i, t-1, 1] * cos(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*sin(2*π*i / 12) + κ_S[1] * s[t])#* s[1][t])
        @constraint(model, [i = 1:6, t = 2:T], γ_star_sto[i, t, 1] == -γ_sto[i, t-1, 1] * sin(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*cos(2*π*i / 12) + κ_S[1] * s[t])#* s[1][t])
        @constraint(model, [t = 1:T], S[t] == sum(γ_sto[i, t, 1]  for i in 1:6))
    end

    @operator(model, log_pdf, 3, logpdf_normal)
    #@variable(model, θ[1:T])
    #@constraint(model, [t=1:T], model[:θ][t] ≤  s[t])
    #@constraint(model, [t=1:T], model[:θ][t] ≤ -s[t])
    @objective(model, Max, sum(log_pdf(model[:params][t],model[:fixed_params][2], y[t]) for t in 1:T));
  
    set_start_value.(model[:RWS][:, 1], round.(initial_values["rws"]["values"]; digits = 5))
    set_start_value.(model[:κ_RWS][1], round.(initial_values["rws"]["κ"]; digits = 5))
    set_start_value.(model[:b][:, 1],  round.(initial_values["slope"]["values"]; digits = 5))
    set_start_value.(model[:κ_b][1], round.(initial_values["slope"]["κ"]; digits = 5))
    set_start_value.(model[:s], round.(UnobservedComponentsGAS.scaled_score.(initial_values["param"][:, 1], initial_values["fixed_param"][1], y, 1.0, 1, 1); digits = 5))
   
    set_start_value.(model[:fixed_params], round.(initial_values["fixed_param"]; digits = 5))

    seasonality_dict, stochastic, stochastic_params = UnobservedComponentsGAS.get_seasonality_dict_and_stochastic([seasonality])
    # Próximas linhas para inicializar apenas os kappas que forem de params com sazo estocástica
    idx_params = sort(findall(i -> i != false, seasonality_dict))        
    idx_params_stochastic = idx_params[findall(stochastic_params .!= false)]
    if !deterministic
        #println("Inicializando sazo estocastica")
        set_start_value.(model[:κ_S][idx_params_stochastic], round.(initial_values["seasonality"]["κ"]; digits = 5))
        set_start_value.(model[:γ_sto][:, :, 1], round.(initial_values["seasonality"]["γ"]; digits = 5))
        set_start_value.(model[:γ_star_sto][:, :, 1], round.(initial_values["seasonality"]["γ_star"]; digits = 5))
        set_start_value.(model[:S], round.(initial_values["seasonality"]["values"]; digits = 5)) 
    else
        #println("Inicializando sazo deterministica")
        # println(round.(initial_values["seasonality"]["γ"]; digits = 5))
        set_start_value.(model[:γ_det][:, 1], round.(initial_values["seasonality"]["γ"]; digits = 5))
        set_start_value.(model[:γ_star_det][:, 1], round.(initial_values["seasonality"]["γ_star"]; digits = 5)) 
        set_start_value.(model[:S], round.(initial_values["seasonality"]["values"]; digits = 5)) 
    end    
    
    return model
end

function define_model_all_variable(y, gas_model, solver, deterministic, initial_values)
    T = length(y)
    model = JuMP.Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 50000)
    set_optimizer_attribute(model, "max_cpu_time", 300.)
    set_optimizer_attribute(model, "tol", 5e-3)

    #criando parametros da distribuição
    @variable(model, params[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RWS[1:T])
    @variable(model, -5 <= κ_RWS[1] <= 5)
    @variable(model, b[1:T])
    @variable(model, -5 <= κ_b[1] <= 5)
    @variable(model, S[1:T])

    # set_start_value.(model[:params], y)

    #parameters = Matrix(undef, T, 2)
    #parameters[:, 1] .= model[:params]
    #parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    #s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution())
    @variable(model, s[1:T])
    # @constraint(model, [t = 1:T], s[t] == y[t] - model[:params][t]) # d = 1
    # @constraint(model, [t = 1:T], s[t] * sqrt(model[:fixed_params][2]) == y[t] - model[:params][t]) # d = 0.5
    # @constraint(model, [t = 1:T], s[t] * model[:fixed_params][2] == y[t] - model[:params][t]) # d = 0
    @operator(model, scaled_score_int, 6, UnobservedComponentsGAS.scaled_score)
    @constraint(model, [t = 1:T], s[t] == scaled_score_int(model[:params][t], model[:fixed_params][2],y[t], 0.5, 1, 1))

    @constraint(model, [t = 2:T], b[t]  == b[t - 1] + κ_b[1] * s[t])
    @constraint(model, [t = 2:T], RWS[t] == RWS[t - 1] + b[t - 1] + κ_RWS[1] * s[t])

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
              
        @constraint(model, [t = 1:T], S[t] == sum(γ_det[i, 1]*cos(2 * π * i * t/12) + 
                                                γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6))
    else
        @variable(model, κ_S[1])
        @constraint(model, [i in 1], -5 ≤ κ_S[1] ≤ 5)    

        @variable(model, γ_sto[1:6, 1:T, 1])
        @variable(model, γ_star_sto[1:6, 1:T, 1])

        @constraint(model, [i = 1:6, t = 2:T], γ_sto[i, t, 1] == γ_sto[i, t-1, 1] * cos(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*sin(2*π*i / 12) + κ_S[1] * s[t])
        @constraint(model, [i = 1:6, t = 2:T], γ_star_sto[i, t, 1] == -γ_sto[i, t-1, 1] * sin(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*cos(2*π*i / 12) + κ_S[1] * s[t])

        @constraint(model, [t = 1:T], S[t] == sum(γ_sto[i, t, 1]  for i in 1:6))
        
    end
    @constraint(model, [t = 2:T], params[t]  == RWS[t] + S[t])

    @operator(model, log_pdf, 3, logpdf_normal)
    #@variable(model, θ[1:T])
    #@constraint(model, [t=1:T], model[:θ][t] ≤  s[t])
    #@constraint(model, [t=1:T], model[:θ][t] ≤ -s[t])
    @objective(model, Max, sum(log_pdf(model[:params][t],model[:fixed_params][2], y[t]) for t in 1:T));
  
    UnobservedComponentsGAS.initialize_components!(model, initial_values, gas_model)
    set_start_value.(model[:S], round.(initial_values["seasonality"]["values"]; digits = 5)) 
    set_start_value.(model[:s], round.(UnobservedComponentsGAS.scaled_score.(initial_values["param"][:, 1], initial_values["fixed_param"][1], y, 1.0, 1, 1); digits = 5))
   
    return model
end

function create_and_optimize(y, gas_model, modelo, deterministic, initial_values, seasonality)

    T = length(y)
    if modelo == 1
        t_create = @elapsed model = define_model_param_variable_dyn_expression(y, Ipopt.Optimizer, deterministic, initial_values)
    elseif modelo == 2
        t_create = @elapsed model = define_model_param_expression_dyn_variable(y, Ipopt.Optimizer, deterministic, initial_values, seasonality)
    else
        t_create  = @elapsed model = define_model_all_variable(y, gas_model, Ipopt.Optimizer, deterministic, initial_values)
    end

    t_optim = @elapsed optimize!(model)
    output = UnobservedComponentsGAS.create_output_fit(model, zeros(T,2), y, missing, missing, gas_model, 0.0)
    rmse_param = sqrt(Metrics.mse(output.fit_in_sample, y))
    println(output.model_status)
    return output, t_create, t_optim, rmse_param, value.(model[:s])
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

df_results = DataFrame([[],[],[], [], [], [], [], [], []],
                     ["serie", "T", "model", "t create", "t optim", "rmse train", "rmse test", "mase test", "status"])

for n in 1:N
    println("   Série = $n")
    y_train = data_dict[n]["train"]
    y_test  = data_dict[n]["test"]
    T       = length(y_train)

    # plot(vcat(y_train, y_test))
    # gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    # initial_values = UnobservedComponentsGAS.create_output_initialization(y_train, missing, gas_model)

    # println("       Modelo = 1")
    # gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    # output1, t_create1, t_optim1, rmse_param1, score1 = create_and_optimize(y_train, gas_model, 1, deterministic, initial_values, seasonality);
    # forec1 = UnobservedComponentsGAS.predict(gas_model, output1, y_train, 18, 500)
    # rmse1 = sqrt(Metrics.mse(forec1["mean"], y_test))
    # mase1 = MASE(y_train, y_test, forec1["mean"])

    # println("       Modelo = 2")
    # gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    # output2, t_create2, t_optim2, rmse_param2, score2 = create_and_optimize(y_train, gas_model, 2, deterministic, initial_values, seasonality);
    # forec2 = UnobservedComponentsGAS.predict(gas_model, output2, y_train, 18, 500)
    # rmse2 = sqrt(Metrics.mse(forec2["mean"], y_test))
    # mase2 = MASE(y_train, y_test, forec2["mean"])
    
    # println("       Modelo = 3")
    # gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    # output3, t_create3, t_optim3, rmse_param3, score3 = create_and_optimize(y_train, gas_model, 3, deterministic, initial_values, seasonality);
    # forec3 = UnobservedComponentsGAS.predict(gas_model, output3, y_train, 18, 500)
    # rmse3 = sqrt(Metrics.mse(forec3["mean"], y_test))
    # mase3 = MASE(y_train, y_test, forec3["mean"])

    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 0.0, "random walk slope", seasonality, missing)
    t_optimp = @elapsed  fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train);
    forecp = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, 18, 500);
    masep = MASE(y_train, y_test, forecp["mean"])
    rmse_paramp = sqrt(Metrics.mse(fitted_model.fit_in_sample, y_train))
    rmsep = sqrt(Metrics.mse(forecp["mean"], y_test))
    
    # plot(vcat(y_train, y_test), label = "observed", color = :black)
    # plot!(vcat(output1.fit_in_sample, ones(18) * NaN), label = "fit1")
    # plot!(vcat(output2.fit_in_sample, ones(18) * NaN), label = "fit2")
    # plot!(vcat(output3.fit_in_sample, ones(18) * NaN), label = "fit3")
    # plot!(vcat(ones(length(y_train)) * NaN,  forec1["mean"]), label = "forec1")
    # plot!(vcat(ones(length(y_train)) * NaN,  forec2["mean"]), label = "forec2")
    # plot!(vcat(ones(length(y_train)) * NaN,  forec3["mean"]), label = "forec3")

    # plot(y_test)
    # plot!(forec1["mean"], label = "forec1")
    # plot!(forec2["mean"], label = "forec2")
    # plot!(forec3["mean"], label = "forec3")

    # push!(df_results, [n, T, "model 1", t_create1, t_optim1, rmse_param1, rmse1, mase1, output1.model_status])
    # push!(df_results, [n, T, "model 2", t_create2, t_optim2, rmse_param2, rmse2, mase2, output2.model_status])
    # push!(df_results, [n, T, "model 3", t_create3, t_optim3, rmse_param3, rmse3, mase3, output3.model_status])
    push!(df_results, [n, T, "package", -1, t_optimp, rmse_paramp, rmsep, masep, fitted_model.model_status])


    CSV.write("results_package_d0.csv", df_results)
end   






# function define_model_all_expression(y, solver)
#     T = length(y)
#     model = JuMP.Model(Ipopt.Optimizer)
#     set_silent(model)
#     set_optimizer_attribute(model, "max_cpu_time",300.0)
#     #criando parametros da distribuição
#     @variable(model, μ1)
#     @variable(model, σ2 ≥ 1e-4)
#     @variable(model, RW1)
#     @variable(model, 5 <= κ_RW <= 5)
#     #@variable(model, b1)
#     @variable(model, 5 <= κ_b <= 5)

#     set_start_value.(model[:μ1], y[1])

#     μ  = Vector(undef, T)
#     RW = Vector(undef, T)
#     #b  = Vector(undef, T)
#     s  = Vector(undef, T)
    
#     parameters = Matrix(undef, T, 2)

#     μ[1]  = model[:μ1]
#     RW[1] = model[:RW1]
#     #b[1]  = model[:b1]
#     parameters[1, :] = [model[:μ1], model[:σ2]]
#     s[1] = 0.0

#     @operator(model, scaled_score_j, 6, UnobservedComponentsGAS.scaled_score)

#     for t in 2:T
#         s[t] = scaled_score_j(parameters[t-1, 1], parameters[t-1, 2], y[t-1], 1.0, 1, 1)
        
#         #RW[t] = RW[t-1]+ model[:κ_RW]*s[t] #+ b[t-1] 
#         RW[t] = RW1 + model[:κ_RW]*sum(s[i] for i in 2:t)  #+ b[t-1] 
#         #b[t]  = b[t-1] + model[:κ_b]*s[t]

#         μ[t] = RW[t]
#         parameters[t, :] = [μ[t], model[:σ2]]
#     end

#     @expression(model, RW, RW);
#     @expression(model, b, b);
#     @expression(model, μ, μ);
#     @expression(model, s, s);

#     @operator(model, log_pdf, 3, logpdf_normal)
#     @time @objective(model, Max, sum(log_pdf(model[:μ][t], model[:σ2], y[t]) for t in 1:T));
#     return model
# end
