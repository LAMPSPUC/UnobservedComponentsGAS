using CSV, Statistics
using DataFrames

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

folder = "results_model1_sm/"
model = "model1_sm"
d = "d1"
data = CSV.read(folder*"results_$(model)_$(d).csv", DataFrame)

results = calculate_statistics(data)
CSV.write(folder*"statistics_$(model)_$(d).csv", results; delim=';', decimal=',')

df_freq_status = get_freq_status(data)
CSV.write(folder*"freq_status_$(model)_$(d).csv", df_freq_status; delim=';', decimal=',')
