using Plots, CSV, DataFrames, MLJBase

import Pkg
Pkg.activate(".")
Pkg.instantiate()
include("src/UnobservedComponentsGAS.jl")

function create_folder_if_not_exists(folder_path::String)
    if !isdir(folder_path)
        mkdir(folder_path)
        println("Folder created: $folder_path")
    else
        println("Folder already exists: $folder_path")
    end
end

path_data = pwd() * "/../../0_Dados/"
path_folders = pwd() * "/StochasticSeasonality/"

series = ["ts_seasonality_var_1", "ts_seasonality_var_100", 
            "ts_seasonality_var_10_irregular_100","ts_seasonality_var_100_level_100"]


for serie in series
    println(serie)
    data = CSV.read(path_data*"$serie.csv", DataFrame)
    for K in 1:2  # Number of harmonics
        println("  K = $K")
        for κ_max in [1,2,3]
            println("    κ_max = $(κ_max)")
            steps_ahead = 12
            y_train     = Float64.(data[1:end-steps_ahead, K])
            y_val       = Float64.(data[end-steps_ahead+1:end, K])

            dist                    = UnobservedComponentsGAS.NormalDistribution()
            time_varying_parameters = [true, false]
            d                       = 1.0
            level                   = ["random walk", ""]
            seasonality             = ["stochastic 12", ""]
            ar                      = [missing, missing]

            model = UnobservedComponentsGAS.GASModel(dist, time_varying_parameters, d, level,seasonality, ar);
            fitted_model = UnobservedComponentsGAS.fit(model, y_train; κ_max = κ_max);

            create_folder_if_not_exists(path_folders * "$serie/")
            create_folder_if_not_exists(path_folders * "$serie/K_$(K)/")
            create_folder_if_not_exists(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/")

            if fitted_model.model_status == "LOCALLY_SOLVED"
                forec        = UnobservedComponentsGAS.predict(model, fitted_model, y_train, steps_ahead, 500);

                df_kappas = DataFrame(Dict("level" =>fitted_model.components["param_1"]["level"]["hyperparameters"]["κ"],
                "seasonality" =>fitted_model.components["param_1"]["seasonality"]["hyperparameters"]["κ"],
                "status" => fitted_model.model_status))
                CSV.write(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/kappas.csv", df_kappas)
        
                UnobservedComponentsGAS.plot_residuals(fitted_model; type="std")
                savefig(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/residuals")

                UnobservedComponentsGAS.plot_acf_residuals(fitted_model; type = "std", lags = 26)
                savefig(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/fac")

                plot(title = "Fit in Sample")
                plot!(y_train[2:end], label = "time series")
                plot!(fitted_model.fit_in_sample[2:end], label = "fit in sample")
                savefig(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/fit")

                plot(title = "Mean Forecast with Confidence Intervals")
                plot!(forec["intervals"]["95"]["lower"], fillrange = forec["intervals"]["95"]["upper"], fillalpha = 0.15, color = :grey, label = "95% Confidence band")
                plot!(forec["intervals"]["80"]["lower"], fillrange = forec["intervals"]["80"]["upper"], fillalpha = 0.15, color = :darkgrey, label = "80% Confidence band")
                plot!(forec["intervals"]["95"]["upper"], label = "", color = :grey)
                plot!(forec["intervals"]["80"]["upper"], label = "", color = :darkgrey)
                plot!(y_val, label = "time series", color = :blue)
                plot!(forec["mean"], label = "mean forecast", color = :red)
                savefig(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/forecast")
            else
                df_kappas = DataFrame(Dict("status" => fitted_model.model_status))
                CSV.write(path_folders * "$serie/K_$(K)/kappa_$(κ_max)/kappas.csv", df_kappas)
            end
        end
    end
end