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
   
    gas_model_lognormal         = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false],0.5, "random walk slope", "deterministic 12", missing)
    gas_model_lognormal_2params = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, true], 0.5, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
   
    gas_model_lognormal_X         = deepcopy(gas_model_lognormal)
    gas_model_lognormal_X_2params = deepcopy(gas_model_lognormal_2params)
   
    model_lognormal, parameters_lognormal, initial_values_lognormal                         = UnobservedComponentsGAS.create_model(gas_model_lognormal, y, missing)
    model_lognormal_2params, parameters_lognormal_2params, initial_values_lognormal_2params = UnobservedComponentsGAS.create_model(gas_model_lognormal_2params, y, missing)
    
    model_lognormal_X, parameters_lognormal_X, initial_values_lognormal_X                         = UnobservedComponentsGAS.create_model(gas_model_lognormal_X, y, X, missing);
    model_lognormal_X_2params, parameters_lognormal_X_2params, initial_values_lognormal_X_2params = UnobservedComponentsGAS.create_model(gas_model_lognormal_X_2params, y, X, missing);
    
    fitted_model_lognormal         = UnobservedComponentsGAS.fit(gas_model_lognormal, y; tol = 5e-2)
    fitted_model_lognormal_2params = UnobservedComponentsGAS.fit(gas_model_lognormal_2params, y; tol = 5e-2)
    fitted_model_lognormal_X         = UnobservedComponentsGAS.fit(gas_model_lognormal_X, y, X)
    fitted_model_lognormal_X_2params = UnobservedComponentsGAS.fit(gas_model_lognormal_X_2params, y, X)

    forecast_lognormal         = UnobservedComponentsGAS.predict(gas_model_lognormal, fitted_model_lognormal, y, steps_ahead, num_scenarious)
    #forecast_lognormal_X       = UnobservedComponentsGAS.predict(gas_model_lognormal_X, fitted_model_lognormal_X, y, X_lognormal_forec, steps_ahead, num_scenarious)
    forecast_lognormal_2params = UnobservedComponentsGAS.predict(gas_model_lognormal_2params, fitted_model_lognormal_2params, y, steps_ahead, num_scenarious)

    scenarios_lognormal         = UnobservedComponentsGAS.simulate(gas_model_lognormal, fitted_model_lognormal, y, steps_ahead, num_scenarious)
    # scenarios_lognormal_X       = UnobservedComponentsGAS.simulate(gas_model_lognormal_X, fitted_model_lognormal_X, y, X_lognormal_forec, steps_ahead, num_scenarious)
    scenarios_lognormal_2params = UnobservedComponentsGAS.simulate(gas_model_lognormal_2params, fitted_model_lognormal_2params, y, steps_ahead, num_scenarious)

    @testset " --- Testing create_model functions" begin
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
    end
    
    @testset " --- Testing fit functions" begin        
        # "Test if termination_status is correct"
        possible_status = ["LOCALLY_SOLVED", "TIME_LIMIT"]
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
        #@test(all(y->y==fitted_model_lognormal.fitted_params["param_2"][1],fitted_model_lognormal.fitted_params["param_2"]))
        @test(!all(y->y==fitted_model_lognormal_2params.fitted_params["param_1"][1],fitted_model_lognormal_2params.fitted_params["param_1"]))
        @test(!all(y->y==fitted_model_lognormal_2params.fitted_params["param_2"][1],fitted_model_lognormal_2params.fitted_params["param_2"]))
        # @test(!all(y->y==fitted_model_lognormal_X.fitted_params["param_1"][1],fitted_model_lognormal_X.fitted_params["param_1"]))
        #@test(all(y->y==fitted_model_lognormal_X.fitted_params["param_2"][1],fitted_model_lognormal_X.fitted_params["param_2"]))
        @test(!all(y->y==fitted_model_lognormal_X_2params.fitted_params["param_1"][1],fitted_model_lognormal_X_2params.fitted_params["param_1"]))
        @test(!all(y->y==fitted_model_lognormal_X_2params.fitted_params["param_2"][1],fitted_model_lognormal_X_2params.fitted_params["param_2"]))
        
        # "Test if all residuals are being generated"
        residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
        @test(all(keys(fitted_model_lognormal.residuals)         .== residuals_types))
        @test(all(keys(fitted_model_lognormal_2params.residuals) .== residuals_types))
    end

    @testset " --- Test forecast function ---" begin
        @test(isapprox(forecast_lognormal["mean"], vec(mean(scenarios_lognormal, dims = 2)); rtol = 1e-3)) 
        @test(size(scenarios_lognormal) == (steps_ahead, num_scenarious))

        @test(isapprox(forecast_lognormal["intervals"]["80"]["lower"], [quantile(scenarios_lognormal[t,:], 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal["intervals"]["80"]["upper"], [quantile(scenarios_lognormal[t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal["intervals"]["95"]["lower"], [quantile(scenarios_lognormal[t,:], 0.05/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal["intervals"]["95"]["upper"], [quantile(scenarios_lognormal[t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

        @test(isapprox(forecast_lognormal_2params["mean"], vec(mean(scenarios_lognormal_2params, dims = 2)); rtol = 1e-3)) 
        @test(size(scenarios_lognormal_2params) == (steps_ahead, num_scenarious))

        @test(isapprox(forecast_lognormal_2params["intervals"]["80"]["lower"], [quantile(scenarios_lognormal_2params[t,:], 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal_2params["intervals"]["80"]["upper"], [quantile(scenarios_lognormal_2params[t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal_2params["intervals"]["95"]["lower"], [quantile(scenarios_lognormal_2params[t,:], 0.05/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_lognormal_2params["intervals"]["95"]["upper"], [quantile(scenarios_lognormal_2params[t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

        # @test(isapprox(forecast_lognormal_X["mean"], vec(mean(forecast_lognormal_X["scenarios"], dims = 2)); rtol = 1e-3)) 
        # @test(size(forecast_lognormal_X["scenarios"]) == (steps_ahead, num_scenarious))

        # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
        # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
        # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))
    end


    @testset " --- Test quality of fit and forecast - LogNormal" begin
        y         = time_series[1:end-steps_ahead,1]
        y_test    = time_series[end-steps_ahead+1:end, 1]

        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, false],
                                                        1.0, "random walk slope", "deterministic 12", missing)
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

        @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-1))
        @test(isapprox(forec["mean"], y_test; rtol = 1e2))
    end

    @testset " --- Test quality of fit - LogNormal with 2 params" begin
        y         = time_series[1:end-steps_ahead,4]
        y_test    = time_series[end-steps_ahead+1:end, 4]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, true],
                                                        0.5, ["random walk slope", "random walk"], ["deterministic 12", "deterministic 12"], [missing, missing])
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

        hcat(fitted_model.fit_in_sample[2:end], y[2:end])
        @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e-1))
        @test(isapprox(forec["mean"], y_test; rtol = 1e2))
    end
            
end