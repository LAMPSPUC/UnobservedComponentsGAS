@testset "Fit & Forecast LogNormal" begin  
    
    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    T,N = size(time_series)
    y = time_series[:,2]
    X = [2*y y/2 rand(T)]
    
    steps_ahead    = 12
    num_scenarious = 500
    X_lognormal_forec     = hcat(y[end-steps_ahead+1:end].+5*rand(steps_ahead), 
                            y[end-steps_ahead+1:end].+10*rand(steps_ahead),
                            rand(steps_ahead))

    function test_initial_values_components(initial_values, rw, rws, ar, seasonality)
        ismissing(seasonality) ? s = false : s = true
        ar == false ? ar_bool = false : ar_bool = true
        dict_has_component = Dict("rws" => rws, "rw" => rw, "ar" => ar_bool, "seasonality" => s)

        dict_tests = Dict()
        for component in keys(initial_values)
            if !occursin("param", component) && component != "explanatories"
                dict_tests[component] = !all(iszero.(initial_values[component]["values"]))
            elseif occursin("param", component) && component != "explanatories"
                dict_tests[component] = !all(iszero.(initial_values[component]))
            end
        end

        return all([dict_tests["rw"] == rw, dict_tests["rws"] == rws, dict_tests["ar"] == ar_bool, dict_tests["seasonality"] == s])
    end

    function convert_dict_keys_to_string(dict)::Dict{String, Any}
        new_dict = Dict{String, Any}()
        for (key, value) in dict
            new_dict[string(key)] = value
        end
        return new_dict
    end

    rw          = false
    rws         = true
    ar          = false
    seasonality = 12    

    dist_lognormal    = UnobservedComponentsGAS.LogNormalDistribution()
   
    gas_model_lognormal         = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_lognormal_2params = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, true], 1.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
   
    gas_model_lognormal_X            = deepcopy(gas_model_lognormal)
    gas_model_lognormal_X_2params    = deepcopy(gas_model_lognormal_2params)
   
    @info(" --- Testing create_model functions")
    # Create model with no explanatory series
    model_lognormal, parameters_lognormal, initial_values_lognormal                         = UnobservedComponentsGAS.create_model(gas_model_lognormal, y, missing)
    model_lognormal_2params, parameters_lognormal_2params, initial_values_lognormal_2params = UnobservedComponentsGAS.create_model(gas_model_lognormal_2params, y, missing)
    
    model_lognormal_X, parameters_lognormal_X, initial_values_lognormal_X                         = UnobservedComponentsGAS.create_model(gas_model_lognormal_X, y, X, missing);
    model_lognormal_X_2params, parameters_lognormal_X_2params, initial_values_lognormal_X_2params = UnobservedComponentsGAS.create_model(gas_model_lognormal_X_2params, y, X, missing);
    
    @test(size(parameters_lognormal)         == (T,2))
    @test(size(parameters_lognormal_2params) == (T,2))
    @test(typeof(model_lognormal)            == JuMP.Model)
    @test(typeof(model_lognormal_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_lognormal, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_lognormal_2params, rw, rws, ar, seasonality))
    
    @test(size(parameters_lognormal_X)         == (T,2))
    @test(size(parameters_lognormal_X_2params) == (T,2))
    @test(typeof(model_lognormal_X)            == JuMP.Model)
    @test(typeof(model_lognormal_X_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_lognormal_X, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_lognormal_X_2params, rw, rws, ar, seasonality))
    
    @info(" --- Testing fit functions")
    fitted_model_lognormal         = UnobservedComponentsGAS.fit(gas_model_lognormal, y)
    fitted_model_lognormal_2params = UnobservedComponentsGAS.fit(gas_model_lognormal_2params, y)
    fitted_model_lognormal_X         = UnobservedComponentsGAS.fit(gas_model_lognormal_X, y, X)
    fitted_model_lognormal_X_2params = UnobservedComponentsGAS.fit(gas_model_lognormal_X_2params, y, X)
    
    # "Test if termination_status is correct"
    possible_status = ["LOCALLY_SOLVED"]
    @test(fitted_model_lognormal.model_status in possible_status)
    @test(fitted_model_lognormal_2params.model_status in possible_status)
    @test(fitted_model_lognormal_X.model_status in possible_status)
    # "Test if selected_variables is missing "
    @test(ismissing(fitted_model_lognormal.selected_variables))
    @test(ismissing(fitted_model_lognormal_2params.selected_variables))
     # "Test if fitted_params has the right keys -> order may be a problem"
    @test(all(keys(fitted_model_lognormal.fitted_params) .== ["param_2","param_1"]))
    @test(all(keys(fitted_model_lognormal_2params.fitted_params) .== ["param_2","param_1"]))
    
    # "Test if all time varying and fixed params are time varying and fixed"
    @test(!all(y->y==fitted_model_lognormal.fitted_params["param_1"][1],fitted_model_lognormal.fitted_params["param_1"]))
    # @test(all(y->y==fitted_model_lognormal.fitted_params["param_2"][1],fitted_model_lognormal.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_lognormal_2params.fitted_params["param_1"][1],fitted_model_lognormal_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_lognormal_2params.fitted_params["param_2"][1],fitted_model_lognormal_2params.fitted_params["param_2"]))
    # @test(!all(y->y==fitted_model_lognormal_X.fitted_params["param_1"][1],fitted_model_lognormal_X.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_lognormal_X.fitted_params["param_2"][1],fitted_model_lognormal_X.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_lognormal_X_2params.fitted_params["param_1"][1],fitted_model_lognormal_X_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_lognormal_X_2params.fitted_params["param_2"][1],fitted_model_lognormal_X_2params.fitted_params["param_2"]))
    
    # "Test if all residuals are being generated"
    residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
    @test(all(keys(fitted_model_lognormal.residuals)         .== residuals_types))
    @test(all(keys(fitted_model_lognormal_2params.residuals) .== residuals_types))

    @info(" --- Test forecast function ---")
    forecast_lognormal         = UnobservedComponentsGAS.predict(gas_model_lognormal, fitted_model_lognormal, y, steps_ahead, num_scenarious)
    # forecast_lognormal_X       = UnobservedComponentsGAS.predict(gas_model_lognormal_X, fitted_model_lognormal_X, y, X_lognormal_forec, steps_ahead, num_scenarious)
    forecast_lognormal_2params = UnobservedComponentsGAS.predict(gas_model_lognormal_2params, fitted_model_lognormal_2params, y, steps_ahead, num_scenarious)


    # @test(isapprox(forecast_lognormal["mean"], vec(mean(forecast_lognormal["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_lognormal["scenarios"]) == (steps_ahead, num_scenarious))

    # @test(isapprox(forecast_lognormal["intervals"]["80"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal["intervals"]["80"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal["intervals"]["95"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal["intervals"]["95"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    # @test(isapprox(forecast_lognormal_2params["mean"], vec(mean(forecast_lognormal_2params["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_lognormal_2params["scenarios"]) == (steps_ahead, num_scenarious))

    # @test(isapprox(forecast_lognormal_2params["intervals"]["80"]["lower"], [quantile(forecast_lognormal_2params["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_2params["intervals"]["80"]["upper"], [quantile(forecast_lognormal_2params["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_2params["intervals"]["95"]["lower"], [quantile(forecast_lognormal_2params["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_2params["intervals"]["95"]["upper"], [quantile(forecast_lognormal_2params["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    # @test(isapprox(forecast_lognormal_X["mean"], vec(mean(forecast_lognormal_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    # @test(size(forecast_lognormal_X["scenarios"]) == (steps_ahead, num_scenarious))

    # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))


    @info(" --- Test quality of fit and forecast - LogNormal")
    N = 10
    T = size(time_series, 1)

    fitted_values = zeros(T-steps_ahead,N)
    forec_values  = zeros(steps_ahead, N)
    y_fitted      = zeros(T-steps_ahead,N)
    y_forec       = zeros(steps_ahead, N)

    for j in 1:N
        y         = time_series[1:end-steps_ahead,j]
        y_test    = time_series[end-steps_ahead+1:end, j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, false],
                                                     1.0, "random walk slope", "deterministic 12", missing)
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

        fitted_values[:,j] .= fitted_model.fit_in_sample
        forec_values[:,j]  .= forec["mean"] 
        y_fitted[:,j]      .= y
        y_forec[:,j]       .= y_test
    end

    @test(isapprox(mean(fitted_values[2:end,:], dims = 2), mean(y_fitted[2:end,:], dims = 2); rtol = 1e-1))
    # @test(isapprox(mean(forec_values[2:end,:], dims = 2), mean(y_forec[2:end,:], dims = 2); rtol = 1e-1))

    @info(" --- Test quality of fit - LogNormal with 2 params")
    fitted_values_2params = zeros(T-steps_ahead,N)
    forec_values_2params  = zeros(steps_ahead, N)
    y_fitted_2params      = zeros(T-steps_ahead,N)
    y_forec_2params       = zeros(steps_ahead, N)

    for j in 1:N
        y         = time_series[1:end-steps_ahead,j]
        y_test    = time_series[end-steps_ahead+1:end, j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, true],
                                                     1.0, ["random walk slope", "random walk"], ["deterministic 12", "deterministic 12"], [missing, missing])
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)
        
        fitted_values_2params[:,j] .= fitted_model.fit_in_sample
        forec_values_2params[:,j]  .= forec["mean"] 
        y_fitted_2params[:,j]      .= y
        y_forec_2params[:,j]       .= y_test
    end

    @test(isapprox(mean(fitted_values_2params[2:end,:], dims = 2), mean(y_fitted_2params[2:end,:], dims = 2); rtol = 1e-1))
    # @test(isapprox(mean(forec_values_2params[2:end,:], dims = 2), mean(y_forec_2params[2:end,:], dims = 2); rtol = 1e-1))

    @info(" --- Test quality of fit - LogNormal with robust")
    # fitted_values = zeros(T-steps_ahead,N)
    # forec_values  = zeros(steps_ahead, N)
    # y_fitted      = zeros(T-steps_ahead,N)
    # y_forec       = zeros(steps_ahead, N)
    
    # for j in 1:N
    #     y         = time_series[1:end-steps_ahead,j]
    #     y_test    = time_series[end-steps_ahead+1:end, j]
    #     gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, false],
    #                                                  1.0, "random walk slope", "deterministic 12", 1)
    #     fitted_model = UnobservedComponentsGAS.fit(gas_model, y; α = 0.0, robust = true)
    #     forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    #     fitted_values[:,j] .= fitted_model.fit_in_sample
    #     forec_values[:,j]  .= forec["mean"] 
    #     y_fitted[:,j]      .= y
    #     y_forec[:,j]       .= y_test
    # end

    # @test(isapprox(mean(fitted_values[2:end,:], dims = 2), mean(y_fitted[2:end,:], dims = 2); rtol = 1e-1))
    # @test(isapprox(mean(forec_values[2:end,:], dims = 2), mean(y_forec[2:end,:], dims = 2); rtol = 1e-1))

end