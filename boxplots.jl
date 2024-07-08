using CSV
using DataFrames
using StatsPlots

# Load the CSV file
data = CSV.read("results_variables_expressions)models23_5000series.csv", DataFrame)

# Function to calculate statistics grouped by the model column
function calculate_statistics(df::DataFrame)
    df = select(df, Not(["T","serie"]))
    grouped_df = groupby(df, :modelo)
    
    results = DataFrame(model=String[], row=Int[], mean=Float64[], std=Float64[], median=Float64[])
    
    for g in grouped_df
        model_name = unique(g.modelo)[1]
        values = Matrix(select(g, Not(:modelo)))
        
        for i in 1:size(values, 1)
            row_values = values[i, :]
            row_mean = mean(row_values)
            row_std = std(row_values)
            row_median = median(row_values)
            push!(results, (model_name, i, row_mean, row_std, row_median))
        end
    end
    
    return results
end

calculate_statistics(data)

# Rename columns to be Julia friendly
rename!(data, Dict(Symbol("t_create") => :t_create))
data[:,:log_t_create] = log.(data[:,:t_create])

rename!(data, Dict(Symbol("t_optim") => :t_optim))
data[:,:log_t_optim] = log.(data[:,:t_optim])

rename!(data, Dict(Symbol("rmse_train") => :rmse_train))
data[:,:log_rmse_train] = log.(data[:,:rmse_train])

rename!(data, Dict(Symbol("rmse_test") => :rmse_test))
data[:,:log_rmse_test] = log.(data[:,:rmse_test])

rename!(data, Dict(Symbol("mape_train") => :mape_train))
data[:,:log_mape_train] = log.(data[:,:mape_train])

rename!(data, Dict(Symbol("mape_test") => :mape_test))
data[:,:log_mape_test] = log.(data[:,:mape_test])


# Create boxplot
box1 = @df data boxplot(:modelo, :log_t_create, legend=false, ylabel="Tempo [log(s)]", title="Boxplot de tempo de criação do modelo")

box2 = @df data boxplot(:modelo, :log_t_optim, legend=false, ylabel="Tempo [log(s)]", title="Boxplot de tempo de otimização do modelo")

box3 = @df data boxplot(:modelo, :log_rmse_train, legend=false, xlabel="Modelo", ylabel="RMSE", title="Boxplot RMSE de treino por modelo")

box4 = @df data boxplot(:modelo, :log_rmse_test, legend=false, xlabel="Modelo", ylabel="RMSE", title="Boxplot RMSE de teste por modelo")

box5 = @df data boxplot(:modelo, :log_mape_train, legend=false, xlabel="Modelo", ylabel="MAPE", title="Boxplot MAPE de treino por modelo")

box6 = @df data boxplot(:modelo, :log_mape_test, legend=false, xlabel="Modelo", ylabel="MAPE", title="Boxplot MAPE de teste por modelo")

plot(box1, box2, box3, box4, box5, box6, layout=(3,2), size=(1500,700), margin = 7Plots.mm)