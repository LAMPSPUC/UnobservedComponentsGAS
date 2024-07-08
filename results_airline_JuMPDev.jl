using CSV, DataFrames, Dates, Plots

# Adicionar o seu pacote


function read_airline(path, steps_ahead)
    serie = CSV.read(path,DataFrame)

    y     = float.(serie[:,2])
    dates = serie[:,1]
    
    len_train   = length(y) - steps_ahead
    
    y_train = y[1:len_train] 
    y_test  = y[len_train+1:end]

    dates_train = dates[1:len_train]
    dates_test  = dates[len_train+1:end]

    return y_train, y_test, dates_train, dates_test
end


function plot_result(dates_train, dates_test, y_train, y_test, 
                    fit_in_sample, forec_mean, forec_ic95_lb, forec_ic95_ub,
                    model)
    
    plot(title = "Fit and Forecast - Airline - $model", titlefontsize = 12)
    plot!(dates_train, y_train, color = "#02376B", label = "series")
    plot!(dates_train, fit_in_sample, color = "#F77805", label = "fit in sample")

    plot!(dates_test, forec_ic95_lb, fillrange = forec_ic95_ub, fillalpha = 0.15, color = :grey, label = "95% CI")
    plot!(dates_test, forec_ic95_ub, label = "", color = :grey)

    plot!(dates_test, y_test, color = "#02376B", label = "")
    plot!(dates_test, forec_mean, color = :red, label = "forecast")

    plot!(xformatter = x -> Dates.format(Date(Dates.UTD(x)), "yyyy-mm"))

end

# Alterar o path abaixo conforme necessario
# path_series  = current_path*"\\..\\..\\0_Dados\\airline.csv"
# current_path = pwd()

steps_ahead = 24
y_train, y_test, dates_train, dates_test = read_airline(path_series, steps_ahead)

#Adicionar cpdigo para estimar e prever o modelo para a série y_train


plot_result(dates_train, dates_test, y_train, y_test, 
            fitted_model.fit_in_sample, forec["mean"], forec["intervals"]["95"]["lower"], forec["intervals"]["95"]["upper"],
            "nome do modelo")
# savefig("gas_airline.png")




