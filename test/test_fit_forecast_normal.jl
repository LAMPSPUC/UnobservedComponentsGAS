@testset "Fit & Forecast Normal" begin  
    
    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_normal_rws_d1.csv"), DataFrame)
    T = size(time_series, 1)
    y = time_series[:,2]
    X = [2*y y/2 rand(T)]
    
    steps_ahead    = 12
    num_scenarious = 500
    X_normal_forec     = hcat(y[end-steps_ahead+1:end].+5*rand(steps_ahead), 
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

    dist_normal    = UnobservedComponentsGAS.NormalDistribution()
   
    gas_model_normal         = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_normal_2params = UnobservedComponentsGAS.GASModel(dist_normal, [true, true], 1.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
   
    gas_model_normal_X            = deepcopy(gas_model_normal)
    gas_model_normal_X_2params    = deepcopy(gas_model_normal_2params)
   
    @info(" --- Testing create_model functions")
    # Create model with no explanatory series
    model_normal, parameters_normal, initial_values_normal                         = UnobservedComponentsGAS.create_model(gas_model_normal, y, missing)
    model_normal_2params, parameters_normal_2params, initial_values_normal_2params = UnobservedComponentsGAS.create_model(gas_model_normal_2params, y, missing)
    
    model_normal_X, parameters_normal_X, initial_values_normal_X                         = UnobservedComponentsGAS.create_model(gas_model_normal_X, y, X, missing);
    model_normal_X_2params, parameters_normal_X_2params, initial_values_normal_X_2params = UnobservedComponentsGAS.create_model(gas_model_normal_X_2params, y, X, missing);
    
    @test(size(parameters_normal)         == (T,2))
    @test(size(parameters_normal_2params) == (T,2))
    @test(typeof(model_normal)            == JuMP.Model)
    @test(typeof(model_normal_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_normal, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_normal_2params, rw, rws, ar, seasonality))
    
    @test(size(parameters_normal_X)         == (T,2))
    @test(size(parameters_normal_X_2params) == (T,2))
    @test(typeof(model_normal_X)            == JuMP.Model)
    @test(typeof(model_normal_X_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_normal_X, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_normal_X_2params, rw, rws, ar, seasonality))
    
    @info(" --- Testing fit functions")
    fitted_model_normal         = UnobservedComponentsGAS.fit(gas_model_normal, y; tol = 5e-2)
    fitted_model_normal_2params = UnobservedComponentsGAS.fit(gas_model_normal_2params, y; tol = 5e-2)
    fitted_model_normal_X         = UnobservedComponentsGAS.fit(gas_model_normal_X, y, X; tol = 5e-2)
    fitted_model_normal_X_2params = UnobservedComponentsGAS.fit(gas_model_normal_X_2params, y, X; tol = 5e-2)
    
    # "Test if termination_status is correct"
    possible_status = ["LOCALLY_SOLVED", "TIME_LIMIT"]
    @test(fitted_model_normal.model_status in possible_status)
    @test(fitted_model_normal_2params.model_status in possible_status)
    @test(fitted_model_normal_X.model_status in possible_status)
    # "Test if selected_variables is missing "
    @test(ismissing(fitted_model_normal.selected_variables))
    @test(ismissing(fitted_model_normal_2params.selected_variables))
     # "Test if fitted_params has the right keys -> order may be a problem"
    @test(all(keys(fitted_model_normal.fitted_params) .== ["param_2","param_1"]))
    @test(all(keys(fitted_model_normal_2params.fitted_params) .== ["param_2","param_1"]))
    
    # "Test if all time varying and fixed params are time varying and fixed"
    @test(!all(y->y==fitted_model_normal.fitted_params["param_1"][1],fitted_model_normal.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_normal.fitted_params["param_2"][1],fitted_model_normal.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_2params.fitted_params["param_1"][1],fitted_model_normal_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_normal_2params.fitted_params["param_2"][1],fitted_model_normal_2params.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_X.fitted_params["param_1"][1],fitted_model_normal_X.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_normal_X.fitted_params["param_2"][1],fitted_model_normal_X.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_X_2params.fitted_params["param_1"][1],fitted_model_normal_X_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_normal_X_2params.fitted_params["param_2"][1],fitted_model_normal_X_2params.fitted_params["param_2"]))
    
    # "Test if all residuals are being generated"
    residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
    @test(all(keys(fitted_model_normal.residuals)         .== residuals_types))
    @test(all(keys(fitted_model_normal_2params.residuals) .== residuals_types))

    @info(" --- Test forecast function ---")
    forecast_normal         = UnobservedComponentsGAS.predict(gas_model_normal, fitted_model_normal, y, steps_ahead, num_scenarious)
    forecast_normal_X       = UnobservedComponentsGAS.predict(gas_model_normal_X, fitted_model_normal_X, y, X_normal_forec, steps_ahead, num_scenarious)
    forecast_normal_2params = UnobservedComponentsGAS.predict(gas_model_normal_2params, fitted_model_normal_2params, y, steps_ahead, num_scenarious)


    @test(isapprox(forecast_normal["mean"], vec(mean(forecast_normal["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal["intervals"]["80"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["80"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @test(isapprox(forecast_normal_2params["mean"], vec(mean(forecast_normal_2params["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal_2params["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal_2params["intervals"]["80"]["lower"], [quantile(forecast_normal_2params["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_2params["intervals"]["80"]["upper"], [quantile(forecast_normal_2params["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_2params["intervals"]["95"]["lower"], [quantile(forecast_normal_2params["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_2params["intervals"]["95"]["upper"], [quantile(forecast_normal_2params["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @test(isapprox(forecast_normal_X["mean"], vec(mean(forecast_normal_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal_X["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal_X["intervals"]["80"]["lower"], [quantile(forecast_normal_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["80"]["upper"], [quantile(forecast_normal_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["95"]["lower"], [quantile(forecast_normal_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["95"]["upper"], [quantile(forecast_normal_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))


    @info(" --- Test quality of fit and forecast - Normal")
    y         = time_series[1:end-steps_ahead,5]
    y_test    = time_series[end-steps_ahead+1:end, 5]
    
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false],
                                                    1.0, "random walk slope", "deterministic 12", missing)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-2))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))

    @info(" --- Test quality of fit - Normal with 2 params")
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, true],
                                                    0.0, ["random walk slope", "random walk"], ["deterministic 12", "deterministic 12"], [missing, missing])
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)
        
    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-1))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))

    @info(" --- Test quality of fit - Normal with robust")
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false],
                                                    1.0, "random walk slope", "deterministic 12", 1)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y; α = 0.0, robust = true)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-2))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))
    

end