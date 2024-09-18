using Plots, CSV, DataFrames, StatsBase, StateSpaceModels

function create_folder_if_not_exists(folder_path::String)
    if !isdir(folder_path)
        mkdir(folder_path)
        println("Folder created: $folder_path")
    else
        println("Folder already exists: $folder_path")
    end
end

function plot_acf(residuals, type)
    acf_values    = autocor(residuals)
    lags          = length(acf_values)
    lag_values    = collect(0:lags-1)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval

    p = plot(title="$type Residuals ACF")
    p = plot!(lag_values,acf_values,seriestype=:stem, label="",xticks=(lag_values,lag_values))
    p = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
    return p
end


path_data = pwd() * "/StochasticSeasonality/"
path_folders = pwd() * "/StochasticSeasonality/"

series = ["ts_seasonality_var_1", "ts_seasonality_var_100", 
            "ts_seasonality_var_10_irregular_100","ts_seasonality_var_100_level_100"]


for serie in series
    println(serie)
    data = CSV.read(path_data*"$serie.csv", DataFrame)
    for K in 1:6  # Number of harmonics
        println("  K = $K")

        steps_ahead = 12
        y_train     = Float64.(data[1:end-steps_ahead, K])
        y_val       = Float64.(data[end-steps_ahead+1:end, K])

        ssm = UnobservedComponents(y_train; trend = "random walk", seasonal = "stochastic 12")
        StateSpaceModels.fit!(ssm)
        kf = kalman_filter(ssm)

        res = StateSpaceModels.get_standard_residuals(kf)
        fit = y_train .- res
        forec = vcat(StateSpaceModels.forecast(ssm, steps_ahead).expected_value...)

        forec_σ       = sqrt.(vec(vcat(StateSpaceModels.forecast(ssm, steps_ahead).covariance...)))
        forec_up_95   = forec .+ 1.96.*forec_σ
        forec_down_95 = forec .- 1.96.*forec_σ
        forec_up_80   = forec .+ 1.28.*forec_σ
        forec_down_80 = forec .- 1.28.*forec_σ

        create_folder_if_not_exists(path_folders * "$serie/")
        create_folder_if_not_exists(path_folders * "$serie/SSM/")

        plot(res[13:end], title = "Std Residuals")
        savefig(path_folders * "$serie/SSM/residuals_$K")

        plot_acf(res[13:end], "Std")
        savefig(path_folders * "$serie/SSM/fac_$K")

        plot(title = "Fit in Sample")
        plot!(y_train[2:end], label = "time series")
        plot!(fit[2:end], label = "fit in sample")
        savefig(path_folders * "$serie/SSM/fit_$K")

        plot(title = "Mean Forecast with Confidence Intervals")
        plot!(forec_down_95, fillrange = forec_up_95, fillalpha = 0.15, color = :grey, label = "95% Confidence band")
        plot!(forec_up_95, label = "", color = :grey)
        plot!(forec_down_80, fillrange = forec_up_80, fillalpha = 0.15, color = :grey, label = "80% Confidence band")
        plot!(forec_up_80, label = "", color = :darkgrey)
        plot!(y_val, label = "time series", color = :blue)
        plot!(forec, label = "mean forecast", color = :red)
        savefig(path_folders * "$serie/SSM/forecast_$K")
    end
end