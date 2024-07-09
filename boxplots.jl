using CSV, Statistics
using DataFrames
using StatsPlots


data = CSV.read("results_variables_expressions_MEB_score_manual_d1.csv", DataFrame)


function calculate_statistics(df::DataFrame)
    df = select(df, Not(["T","serie", "status"]))
    grouped_df = groupby(df, :model)
    
    results = DataFrame(model=String[], metric=String[], mean=Float64[], std=Float64[], median=Float64[])
    
    for g in grouped_df
        model_name = unique(g.model)[1]
        values = Matrix(select(g, Not(:model)))
        metrics = names(g)
        for j in 1:size(values, 2)
            metric = metrics[j+1]
            col = values[:, j]
            col_mean = mean(col)
            col_std = std(col)
            col_median = median(col)
            push!(results, (model_name, metric, col_mean, col_std, col_median))
        end
    end
    
    return results
end

function get_freq_status(df)
    df_aux = combine(groupby(df, [:model, :status]), nrow => :count)
    df_aux[:, :freq] = df_aux[:, :count] ./ 100 
    return df_aux
end


results = calculate_statistics(data)
CSV.write("statistics_results_expressions_score_manual_d1.csv", results; delim=';', decimal=',')

df_freq_status = get_freq_status(data)
CSV.write("freq_status_expressions_score_manual_d1.csv", df_freq_status; delim=';', decimal=',')

println("Tempo total = ",(sum(data[:, "t_create"]) + sum(data[:, "t_optim"])) / 60, " seg")

# Rename columns to be Julia friendly
rename!(data, Dict(Symbol("t create") => :t_create))
data[:,:log_t_create] = log.(data[:,:t_create])

rename!(data, Dict(Symbol("t optim") => :t_optim))
data[:,:log_t_optim] = log.(data[:,:t_optim])

rename!(data, Dict(Symbol("rmse train") => :rmse_train))
data[:,:log_rmse_train] = log.(data[:,:rmse_train])

rename!(data, Dict(Symbol("rmse test") => :rmse_test))
data[:,:log_rmse_test] = log.(data[:,:rmse_test])

rename!(data, Dict(Symbol("mase test") => :mase_test))
data[:,:log_mase_test] = log.(data[:,:mase_test])

# Create boxplot
box1 = @df data boxplot(:model, :log_t_create, legend=false, ylabel="Tempo [log(s)]", title="Boxplot de tempo de criação do modelo")
box2 = @df data boxplot(:model, :log_t_optim, legend=false, ylabel="Tempo [log(s)]", title="Boxplot de tempo de otimização do modelo")
box3 = @df data boxplot(:model, :log_rmse_train, legend=false, xlabel="Modelo", ylabel="RMSE", title="Boxplot RMSE de treino por modelo")
box4 = @df data boxplot(:model, :log_rmse_test, legend=false, xlabel="Modelo", ylabel="RMSE", title="Boxplot RMSE de teste por modelo")
box5 = @df data boxplot(:model, :log_mase_test, legend=false, xlabel="Modelo", ylabel="MASE", title="Boxplot MASE de teste por modelo")

plot(box1, box2, box3, box4, box5, layout=(3,2), size=(1500,700), margin = 7Plots.mm)