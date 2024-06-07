@testset "Fit & Forecast tDist" begin  
    
    time_series = CSV.read(joinpath(@__DIR__, "test/data/timeseries_t_rws_d1.csv"), DataFrame)
    T,N = size(time_series)
    y = time_series[:,2]
    X = [2*y y/2 rand(T)]
    
    steps_ahead    = 12
    num_scenarious = 500
    X_t_forec     = hcat(y[end-steps_ahead+1:end].+5*rand(steps_ahead), 
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
    ν           = 5   

    dist_t    = UnobservedComponentsGAS.tLocationScaleDistribution()
   
    gas_model_t         = UnobservedComponentsGAS.GASModel(dist_t, [true, false, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_t_2params = UnobservedComponentsGAS.GASModel(dist_t, [true, true, false], 1.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing, missing])
   
    gas_model_t_X            = deepcopy(gas_model_t)
    gas_model_t_X_2params    = deepcopy(gas_model_t_2params)
   
    @info(" --- Testing create_model functions")
    # Create model with no explanatory series
    model_t, parameters_t, initial_values_t                         = UnobservedComponentsGAS.create_model(gas_model_t, y, ν)
    model_t_2params, parameters_t_2params, initial_values_t_2params = UnobservedComponentsGAS.create_model(gas_model_t_2params, y,  ν)
    
    model_t_X, parameters_t_X, initial_values_t_X                         = UnobservedComponentsGAS.create_model(gas_model_t_X, y, X,  ν);
    model_t_X_2params, parameters_t_X_2params, initial_values_t_X_2params = UnobservedComponentsGAS.create_model(gas_model_t_X_2params, y, X,  ν);
    
    @test(size(parameters_t)         == (T,3))
    @test(size(parameters_t_2params) == (T,3))
    @test(typeof(model_t)            == JuMP.Model)
    @test(typeof(model_t_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_t, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_t_2params, rw, rws, ar, seasonality))
    
    @test(size(parameters_t_X)         == (T,3))
    @test(size(parameters_t_X_2params) == (T,3))
    @test(typeof(model_t_X)            == JuMP.Model)
    @test(typeof(model_t_X_2params)    == JuMP.Model)
    @test(test_initial_values_components(initial_values_t_X, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_t_X_2params, rw, rws, ar, seasonality))
    
    @info(" --- Testing fit functions")
    fitted_model_t         = UnobservedComponentsGAS.fit(gas_model_t, y; tol = 5e-2)
    fitted_model_t_2params = UnobservedComponentsGAS.fit(gas_model_t_2params, y; tol = 1e-2)
    fitted_model_t_X         = UnobservedComponentsGAS.fit(gas_model_t_X, y, X; tol = 5e-2)
    fitted_model_t_X_2params = UnobservedComponentsGAS.fit(gas_model_t_X_2params, y, X; tol = 5e-2)
    
    # "Test if termination_status is correct"
    possible_status = ["LOCALLY_SOLVED"]
    @test(fitted_model_t.model_status in possible_status)
    @test(fitted_model_t_2params.model_status in possible_status)
    @test(fitted_model_t_X.model_status in possible_status)
    # "Test if selected_variables is missing "
    @test(ismissing(fitted_model_t.selected_variables))
    @test(ismissing(fitted_model_t_2params.selected_variables))
     # "Test if fitted_params has the right keys -> order may be a problem"
    @test(all(keys(fitted_model_t.fitted_params) .== ["param_2","param_1","param_3"]))
    @test(all(keys(fitted_model_t_2params.fitted_params) .== ["param_2","param_1","param_3"]))
    
    # "Test if all time varying and fixed params are time varying and fixed"
    @test(!all(y->y==fitted_model_t.fitted_params["param_1"][1],fitted_model_t.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_t.fitted_params["param_2"][1],fitted_model_t.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_t_2params.fitted_params["param_1"][1],fitted_model_t_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_t_2params.fitted_params["param_2"][1],fitted_model_t_2params.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_t_X.fitted_params["param_1"][1],fitted_model_t_X.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_t_X.fitted_params["param_2"][1],fitted_model_t_X.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_t_X_2params.fitted_params["param_1"][1],fitted_model_t_X_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_t_X_2params.fitted_params["param_2"][1],fitted_model_t_X_2params.fitted_params["param_2"]))
    
    # "Test if all residuals are being generated"
    residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
    @test(all(keys(fitted_model_t.residuals)         .== residuals_types))
    @test(all(keys(fitted_model_t_2params.residuals) .== residuals_types))

    @info(" --- Test forecast function ---")
    forecast_t         = UnobservedComponentsGAS.predict(gas_model_t, fitted_model_t, y, steps_ahead, num_scenarious)
    forecast_t_X       = UnobservedComponentsGAS.predict(gas_model_t_X, fitted_model_t_X, y, X_t_forec, steps_ahead, num_scenarious)
    forecast_t_2params = UnobservedComponentsGAS.predict(gas_model_t_2params, fitted_model_t_2params, y, steps_ahead, num_scenarious)


    @test(isapprox(forecast_t["mean"], vec(mean(forecast_t["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_t["intervals"]["80"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t["intervals"]["80"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t["intervals"]["95"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t["intervals"]["95"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @test(isapprox(forecast_t_2params["mean"], vec(mean(forecast_t_2params["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t_2params["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_t_2params["intervals"]["80"]["lower"], [quantile(forecast_t_2params["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_2params["intervals"]["80"]["upper"], [quantile(forecast_t_2params["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_2params["intervals"]["95"]["lower"], [quantile(forecast_t_2params["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_2params["intervals"]["95"]["upper"], [quantile(forecast_t_2params["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @test(isapprox(forecast_t_X["mean"], vec(mean(forecast_t_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t_X["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_t_X["intervals"]["80"]["lower"], [quantile(forecast_t_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_X["intervals"]["80"]["upper"], [quantile(forecast_t_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_X["intervals"]["95"]["lower"], [quantile(forecast_t_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_t_X["intervals"]["95"]["upper"], [quantile(forecast_t_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))


    @info(" --- Test quality of fit and forecast - tDist")
    y         = time_series[1:end-steps_ahead,1]
    y_test    = time_series[end-steps_ahead+1:end, 1]

    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.tLocationScaleDistribution(), [true, false, false],
                                                    1.0, "random walk slope", "deterministic 12", missing)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-2))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))

    @info(" --- Test quality of fit - tDist with 2 params")
    
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.tLocationScaleDistribution(), [true, true, false],
                                                    0.0, ["random walk slope", "random walk", ""], ["deterministic 12", "deterministic 12", ""], [missing, missing, missing])
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-1))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))


    @info(" --- Test quality of fit - tDist with robust")
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.tLocationScaleDistribution(), [true, false, false],
                                                    1.0, "random walk slope", "deterministic 12", 1)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y; α = 0.0, robust = true)
    forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-2))
    @test(isapprox(forec["mean"], y_test; rtol = 1e2))

end

