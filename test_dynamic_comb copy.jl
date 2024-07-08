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
        y_raw = Vector(df_train[i, :])[2:end]
        y_train_raw = y_raw[1:findlast(i->!ismissing(i), y_raw)]
        T = length(y_train_raw)
        y_train = y_train_raw
        y_test  = Vector(df_test[i, :])[2:end]

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

function create_ts_seasonal(T)
    σ2ϵ = 0.8  # Variance of the observation noise
    σ2η = 0.5  # Variance of the level noise
    σ2κ = 0.05  # Variance of the slope noise
    σ2ζ = 0.5  # Variance of the seasonal noise
    
    # Initialization
    β = zeros(T+1)
    μ = zeros(T+1)
    S1 = zeros(T+1)
    C1 = zeros(T+1)
    S2 = zeros(T+1)
    C2 = zeros(T+1)
    y = zeros(T)
    
    # Initial values for level, slope, and seasonal components
    μ[1] = rand(Uniform(0, 10))
    β[1] = rand(Uniform(-1, 1))
    S1[1] = rand(Uniform(-1, 1))
    C1[1] = rand(Uniform(-1, 1))
    S2[1] = rand(Uniform(-1, 1))
    C2[1] = rand(Uniform(-1, 1))
    
    # Time series generation
    for t in 1:T
        seasonal_component = S1[t] * sin(2π*t/T) + C1[t] * cos(2π*t/T) + S2[t] * sin(4π*t/T) + C2[t] * cos(4π*t/T)
        y[t] = μ[t] + seasonal_component + rand(Normal(0, σ2ϵ))
        
        # Update level and slope
        μ[t+1] = μ[t] + β[t] + rand(Normal(0, σ2η))
        β[t+1] = β[t] + rand(Normal(0, σ2κ))
        
        # Update seasonal components
        S1[t+1] = S1[t] + rand(Normal(0, σ2ζ))
        C1[t+1] = C1[t] + rand(Normal(0, σ2ζ))
        S2[t+1] = S2[t] + rand(Normal(0, σ2ζ))
        C2[t+1] = C2[t] + rand(Normal(0, σ2ζ))
    end
    
    # Output the generated time series
    return y
end

function create_ts(T)
    β = zeros(T+1)
    μ = zeros(T+1)

    σ2ϵ  = 0.8
    σ2η  = 0.5
    σ2κ  = 0.05

    y = zeros(T)

    μ[1] = rand(Uniform(0,10))
    β[1] = rand(Uniform(-1,1))

    for t in 1:T
        y[t]   = μ[t] + rand(Normal(0,σ2ϵ)) 
        μ[t+1] = μ[t] + β[t] + rand(Normal(0,σ2η))
        β[t+1] = β[t] + rand(Normal(0,σ2κ))
    end

    for t in 1:T
        y[t] = y[t] + 4*sin(t*π/6) + rand(Normal(0,0.5)) 
    end
    return y
end

function define_model_param_variable_dyn_expression(y, solver, deterministic)
    T = length(y)
    model = JuMP.Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_cpu_time",300.0)

    #criando parametros da distribuição
    @variable(model, params[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RW1)
    @variable(model, -2 <= κ_RWS[1] <= 2)
    @variable(model, b1)
    @variable(model, -2 <= κ_b[1] <= 2)
    @variable(model, S1)

    set_start_value.(model[:params], y)

    parameters = Matrix(undef, T, 2)
    parameters[:, 1] .= model[:params]
    parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution())

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
    else
        @variable(model, -2 <= κ_S[1] <= 2)
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
        RWS[t] = RWS[t-1] + b[t-1] + model[:κ_RWS][1]*s[1][t]
        b[t]  = b[t-1] + model[:κ_b][1]*s[1][t]
        if deterministic
            S[t] =  sum(γ_det[i, 1]*cos(2 * π * i * t/12) + γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6)
        else
            for i in 1:6
                γ_sto[i, t]      = γ_sto[i, t-1] * cos(2*π*i / 12)  + γ_star_sto[i,t-1]*sin(2*π*i / 12) + κ_S[1] * s[1][t]
                γ_star_sto[i, t] = -γ_sto[i, t-1] * sin(2*π*i / 12) + γ_star_sto[i,t-1]*cos(2*π*i / 12) + κ_S[1] * s[1][t]
            end
            S[t] =  sum(γ_sto[i, t]  for i in 1:6)
        end
    end

    @expression(model, RWS, RWS)
    @expression(model, b, b);
    @expression(model, S, S);
    @constraint(model, [t in 1:T], params[t] == RWS[t] + S[t]);

    @operator(model, log_pdf, 3, logpdf_normal)
    @objective(model, Max, sum(log_pdf(parameters[t, 1],parameters[t, 2], y[t]) for t in 1:T));
    return model
end

function define_model_param_expression_dyn_variable(y, solver, deterministic, initial_values, seasonality)
    T = length(y)
    model = JuMP.Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 300.)
    set_optimizer_attribute(model, "tol", 5e-3)
    #criando parametros da distribuição
    #@variable(model, μ[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RWS[1:T])
    @variable(model, -2 <= κ_RWS[1] <= 2)
    @variable(model, b[1:T])
    @variable(model, -2 <= κ_b[1] <= 2)
    @variable(model, S[1:T])

    @expression(model, params, RWS + S)

    parameters = Matrix(undef, T, 2)
    parameters[:, 1] .= model[:params]
    parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution());

    @constraint(model, [t = 2:T], b[t] == b[t - 1] + κ_b[1] * s[1][t])
    @constraint(model, [t = 2:T], RWS[t] == RWS[t - 1] + b[t - 1] + κ_RWS[1] * s[1][t])

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
        @constraint(model, [t = 1:T], S[t] == sum(γ_det[i, 1]*cos(2 * π * i * t/12) + 
                                                γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6))
    else
        @variable(model, κ_S[1])
        @constraint(model, [i in 1], -2 ≤ κ_S[1] ≤ 2)    

        @variable(model, γ_sto[1:6, 1:T, 1])
        @variable(model, γ_star_sto[1:6, 1:T, 1])

        @constraint(model, [i = 1:6, t = 2:T], γ_sto[i, t, 1] == γ_sto[i, t-1, 1] * cos(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*sin(2*π*i / 12) + κ_S[1] * s[1][t])
        @constraint(model, [i = 1:6, t = 2:T], γ_star_sto[i, t, 1] == -γ_sto[i, t-1, 1] * sin(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*cos(2*π*i / 12) + κ_S[1] * s[1][t])
        @constraint(model, [t = 1:T], S[t] == sum(γ_sto[i, t, 1]  for i in 1:6))
    end

    @operator(model, log_pdf, 3, logpdf_normal)
    @objective(model, Max, sum(log_pdf(parameters[t, 1],parameters[t, 2], y[t]) for t in 1:T));

    set_start_value.(model[:RWS][:, 1], round.(initial_values["rws"]["values"]; digits = 5))
    set_start_value.(model[:κ_RWS][1], round.(initial_values["rws"]["κ"]; digits = 5))
    set_start_value.(model[:b][:, 1],  round.(initial_values["slope"]["values"]; digits = 5))
    set_start_value.(model[:κ_b][1], round.(initial_values["slope"]["κ"]; digits = 5))

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
    set_optimizer_attribute(model, "max_iter", 30000)
    set_optimizer_attribute(model, "max_cpu_time", 300.)
    set_optimizer_attribute(model, "tol", 5e-3)

    #criando parametros da distribuição
    @variable(model, params[1:T])
    @variable(model, fixed_params[2] ≥ 1e-4)
    @variable(model, RWS[1:T])
    @variable(model, -2 <= κ_RWS[1] <= 2)
    @variable(model, b[1:T])
    @variable(model, -2 <= κ_b[1] <= 2)
    @variable(model, S[1:T])

    parameters = Matrix(undef, T, 2)
    parameters[:, 1] .= model[:params]
    parameters[:, 2] .= model[:fixed_params][2]

    #criando variaveis da dinamica
    s = UnobservedComponentsGAS.compute_score(model, parameters, y, 1.0, [true, false], T, UnobservedComponentsGAS.NormalDistribution())

    @constraint(model, [t = 2:T], b[t]  == b[t - 1] + κ_b[1] * s[1][t])
    @constraint(model, [t = 2:T], RWS[t] == RWS[t - 1] + b[t - 1] + κ_RWS[1] * s[1][t])

    if deterministic
        @variable(model, γ_det[1:6, 1])
        @variable(model, γ_star_det[1:6, 1])
              
        @constraint(model, [t = 1:T], S[t] == sum(γ_det[i, 1]*cos(2 * π * i * t/12) + 
                                                γ_star_det[i, 1] * sin(2 * π * i* t/12) for i in 1:6))
    else
        @variable(model, κ_S[1])
        @constraint(model, [i in 1], -2 ≤ κ_S[1] ≤ 2)    

        @variable(model, γ_sto[1:6, 1:T, 1])
        @variable(model, γ_star_sto[1:6, 1:T, 1])

        @constraint(model, [i = 1:6, t = 2:T], γ_sto[i, t, 1] == γ_sto[i, t-1, 1] * cos(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*sin(2*π*i / 12) + κ_S[1] * s[1][t])
        @constraint(model, [i = 1:6, t = 2:T], γ_star_sto[i, t, 1] == -γ_sto[i, t-1, 1] * sin(2*π*i / 12) + 
                                                                                    γ_star_sto[i,t-1, 1]*cos(2*π*i / 12) + κ_S[1] * s[1][t])

        @constraint(model, [t = 1:T], S[t] == sum(γ_sto[i, t, 1]  for i in 1:6))
        
    end
    @constraint(model, [t = 2:T], params[t]  == RWS[t] + S[t])

    @operator(model, log_pdf, 3, logpdf_normal)
    @objective(model, Max, sum(log_pdf(parameters[t, 1],parameters[t, 2], y[t]) for t in 1:T));

    UnobservedComponentsGAS.initialize_components!(model, initial_values, gas_model)
    set_start_value.(model[:S], round.(initial_values["seasonality"]["values"]; digits = 5)) 

    return model
end

function create_and_optimize(y, gas_model, modelo, deterministic, initial_values, seasonality)

    T = length(y)
    if modelo == 1
        t_create = @elapsed model = define_model_param_variable_dyn_expression(y, Ipopt.Optimizer, deterministic)
    elseif modelo == 2
        t_create = @elapsed model = define_model_param_expression_dyn_variable(y, Ipopt.Optimizer, deterministic, initial_values, seasonality)
    else
        t_create  = @elapsed model = define_model_all_variable(y, gas_model, Ipopt.Optimizer, deterministic, initial_values)
    end

    t_optim = @elapsed optimize!(model)
    output = UnobservedComponentsGAS.create_output_fit(model, zeros(T,2), y, missing, missing, gas_model, 0.0)
    rmse_param = sqrt(Metrics.mse(output.fit_in_sample, y))
    mape_param = mape(y, output.fit_in_sample)

    return output, t_create, t_optim, rmse_param, mape_param
end

function mape(y_true, y_pred)
    return mean(abs.((y_true .- y_pred) ./ y_true)) * 100
end

granularity = "Monthly"
N = 5000
data_dict  = build_train_test_dict(read_dataframes(granularity)..., N);

#T_values = [74, 124, 224, 524]
#N        = 10
timeout = 300
deterministic = true
deterministic ? seasonality = "deterministic 12" : seasonality = "stochastic 12"

df_results = DataFrame([[],[],[], [], [], [], [], [], []], ["serie", "T", "modelo", "t_create", "t_optim", "rmse_train", "rmse_test", "mape_train", "mape_test"])
#for T in T_values
    # println("Tamanho = $T")
for n in 1:N
    n % 100 == 0 ? println("   Série = $n") : nothing
    y_train = data_dict[n]["train"]
    y_test  = data_dict[n]["test"]
    T = length(y_train)

    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    initial_values = UnobservedComponentsGAS.create_output_initialization(y_train, missing, gas_model)

    # # println("       Modelo = 1")
    # gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    # output1, t_create1, t_optim1, rmse_param1, mape_param1 = create_and_optimize(y_train, gas_model, 1, deterministic, initial_values, seasonality)
    # forec1 = UnobservedComponentsGAS.predict(gas_model, output1, y_train, length(y_test), 500)
    # rmse1 = sqrt(Metrics.mse(forec1["mean"], y_test))
    # mape1 = mape(y_test, forec1["mean"])
    
    # println("       Modelo = 2")
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    output2, t_create2, t_optim2, rmse_param2, mape_param2 = create_and_optimize(y_train, gas_model, 2, deterministic, initial_values, seasonality)
    forec2 = UnobservedComponentsGAS.predict(gas_model, output2, y_train, length(y_test), 500)
    rmse2 = sqrt(Metrics.mse(forec2["mean"], y_test))
    mape2 = mape(y_test, forec2["mean"])

    # Y[T][n]["output2"] = output2
    # Y[T][n]["forec2"] = forec2

    # plot(y_train)
    # plot!(output2.fit_in_sample)

    # plot(y_test)
    # plot!(forec2["mean"])
    
    # println("       Modelo = 3")
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false], 1.0, "random walk slope", seasonality, missing)
    output3, t_create3, t_optim3, rmse_param3, mape_param3 = create_and_optimize(y_train, gas_model, 3, deterministic, initial_values, seasonality)
    forec3 = UnobservedComponentsGAS.predict(gas_model, output3, y_train, length(y_test), 500)
    rmse3 = sqrt(Metrics.mse(forec3["mean"], y_test))
    mape3 = mape(y_test, forec3["mean"])
    
    # Y[T][n]["output3"] = output3
    # Y[T][n]["forec3"] = forec3

    # plot(y_train)
    # plot!(output3.fit_in_sample)

    # plot(y_test)
    # plot!(forec3["mean"])

    # push!(df_results, [n, T, "model 1", t_create1, t_optim1, rmse_param1, rmse1])
    # push!(df_results, [n, T, "modelo 1", t_create1, t_optim1, rmse_param1, rmse1, mape_param1, mape1])
    push!(df_results, [n, T, "modelo 2", t_create2, t_optim2, rmse_param2, rmse2, mape_param2, mape2])
    push!(df_results, [n, T, "modelo 3", t_create3, t_optim3, rmse_param3, rmse3, mape_param3, mape3])
  CSV.write("results_variables_expressions.csv", df_results)
end   
#end



model = Model(Ipopt.Optimizer)
@variable(model, α1)
@variable(model, -1 ≤ ϕ ≤ 1)
exp = Vector(undef, 5)
exp[1] = model[:α1]
for i in 1:4
    exp[i+1] = model[:ϕ] * exp[i]
end
println(model)

model = Model(Ipopt.Optimizer)
@variable(model, α[1:5])
@variable(model, -1 ≤ ϕ ≤ 1)

for i in 1:4
    @constraint(model, model[:α][i+1] == model[:ϕ] * model[:α][i])
end

α[2] == ϕ*α[1]
α[3] == ϕ*α[2]
α[4] == ϕ*α[3]
α[5] == ϕ*α[4]


print(model)
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
