@testset "Fit & Forecast Gamma" begin  
    
    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    T = size(time_series, 2)
    y = time_series[:,2]
    X = [2*y y/2 rand(T)]
    
    steps_ahead    = 12
    num_scenarious = 500
    X_gamma_forec     = hcat(y[end-steps_ahead+1:end].+5*rand(steps_ahead), 
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

    dist_gamma    = UnobservedComponentsGAS.GammaDistribution()
   
    gas_model_gamma         = UnobservedComponentsGAS.GASModel(dist_gamma, [true, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_gamma_2params = UnobservedComponentsGAS.GASModel(dist_gamma, [true, true], 1.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
   
    gas_model_gamma_X            = deepcopy(gas_model_gamma)
    gas_model_gamma_X_2params    = deepcopy(gas_model_gamma_2params)

    model_gamma, parameters_gamma, initial_values_gamma                         = UnobservedComponentsGAS.create_model(gas_model_gamma, y, missing)
    model_gamma_2params, parameters_gamma_2params, initial_values_gamma_2params = UnobservedComponentsGAS.create_model(gas_model_gamma_2params, y, missing)
        
    model_gamma_X, parameters_gamma_X, initial_values_gamma_X                         = UnobservedComponentsGAS.create_model(gas_model_gamma_X, y, X, missing);
    model_gamma_X_2params, parameters_gamma_X_2params, initial_values_gamma_X_2params = UnobservedComponentsGAS.create_model(gas_model_gamma_X_2params, y, X, missing);
   
    fitted_model_gamma         = UnobservedComponentsGAS.fit(gas_model_gamma, y; tol = 5e-2)
    fitted_model_gamma_2params = UnobservedComponentsGAS.fit(gas_model_gamma_2params, y; tol = 5e-2)
    fitted_model_gamma_X         = UnobservedComponentsGAS.fit(gas_model_gamma_X, y, X; tol = 5e-2)
    fitted_model_gamma_X_2params = UnobservedComponentsGAS.fit(gas_model_gamma_X_2params, y, X; tol = 5e-2)

    forecast_gamma         = UnobservedComponentsGAS.predict(gas_model_gamma, fitted_model_gamma, y, steps_ahead, num_scenarious)
    forecast_gamma_X       = UnobservedComponentsGAS.predict(gas_model_gamma_X, fitted_model_gamma_X, y, X_gamma_forec, steps_ahead, num_scenarious)
    forecast_gamma_2params = UnobservedComponentsGAS.predict(gas_model_gamma_2params, fitted_model_gamma_2params, y, steps_ahead, num_scenarious)

    scenarios_gamma         = UnobservedComponentsGAS.simulate(gas_model_gamma, fitted_model_gamma, y, steps_ahead, num_scenarious)
    scenarios_gamma_X       = UnobservedComponentsGAS.simulate(gas_model_gamma_X, fitted_model_gamma_X, y, X_gamma_forec, steps_ahead, num_scenarious)
    scenarios_gamma_2params = UnobservedComponentsGAS.simulate(gas_model_gamma_2params, fitted_model_gamma_2params, y, steps_ahead, num_scenarious)
    
    @testset "create_model" begin
    # Create model with no explanatory series
        
        @test(size(parameters_gamma)         == (T,2))
        @test(size(parameters_gamma_2params) == (T,2))
        @test(typeof(model_gamma)            == JuMP.Model)
        @test(typeof(model_gamma_2params)    == JuMP.Model)
        @test(test_initial_values_components(initial_values_gamma, rw, rws, ar, seasonality))
        @test(test_initial_values_components(initial_values_gamma_2params, rw, rws, ar, seasonality))
        
        @test(size(parameters_gamma_X)         == (T,2))
        @test(size(parameters_gamma_X_2params) == (T,2))
        @test(typeof(model_gamma_X)            == JuMP.Model)
        @test(typeof(model_gamma_X_2params)    == JuMP.Model)
        @test(test_initial_values_components(initial_values_gamma_X, rw, rws, ar, seasonality))
        @test(test_initial_values_components(initial_values_gamma_X_2params, rw, rws, ar, seasonality))
    end
    
    #@info(" --- Testing fit functions")
    @testset "fit" begin
        # "Test if termination_status is correct"
        possible_status = ["LOCALLY_SOLVED", "TIME_LIMIT"]
        @test(fitted_model_gamma.model_status in possible_status)
        @test(fitted_model_gamma_2params.model_status in possible_status)
        @test(fitted_model_gamma_X.model_status in possible_status)
        # "Test if selected_variables is missing "
        @test(ismissing(fitted_model_gamma.selected_variables))
        @test(ismissing(fitted_model_gamma_2params.selected_variables))
        # "Test if fitted_params has the right keys -> order may be a problem"
        @test(all(keys(fitted_model_gamma.fitted_params) .== ["param_2","param_1"]))
        @test(all(keys(fitted_model_gamma_2params.fitted_params) .== ["param_2","param_1"]))
        
        # "Test if all time varying and fixed params are time varying and fixed"
        @test(!all(y->y==fitted_model_gamma.fitted_params["param_1"][1],fitted_model_gamma.fitted_params["param_1"]))
        @test(all(y->y==fitted_model_gamma.fitted_params["param_2"][1],fitted_model_gamma.fitted_params["param_2"]))
        @test(!all(y->y==fitted_model_gamma_2params.fitted_params["param_1"][1],fitted_model_gamma_2params.fitted_params["param_1"]))
        @test(!all(y->y==fitted_model_gamma_2params.fitted_params["param_2"][1],fitted_model_gamma_2params.fitted_params["param_2"]))
        @test(!all(y->y==fitted_model_gamma_X.fitted_params["param_1"][1],fitted_model_gamma_X.fitted_params["param_1"]))
        @test(all(y->y==fitted_model_gamma_X.fitted_params["param_2"][1],fitted_model_gamma_X.fitted_params["param_2"]))
        @test(!all(y->y==fitted_model_gamma_X_2params.fitted_params["param_1"][1],fitted_model_gamma_X_2params.fitted_params["param_1"]))
        @test(!all(y->y==fitted_model_gamma_X_2params.fitted_params["param_2"][1],fitted_model_gamma_X_2params.fitted_params["param_2"]))
        
        # "Test if all residuals are being generated"
        residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
        @test(all(keys(fitted_model_gamma.residuals)         .== residuals_types))
        @test(all(keys(fitted_model_gamma_2params.residuals) .== residuals_types))
    end

    #@info(" --- Test forecast function ---")
    @testset "forecast" begin
        
        @test(isapprox(forecast_gamma["mean"], vec(mean(scenarios_gamma, dims = 2)); rtol = 1e-3)) 
        @test(size(scenarios_gamma) == (steps_ahead, num_scenarious))

        @test(isapprox(forecast_gamma["intervals"]["80"]["lower"], [quantile(scenarios_gamma[t,:], 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma["intervals"]["80"]["upper"], [quantile(scenarios_gamma[t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma["intervals"]["95"]["lower"], [quantile(scenarios_gamma[t,:], 0.05/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma["intervals"]["95"]["upper"], [quantile(scenarios_gamma[t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

        @test(isapprox(forecast_gamma_2params["mean"], vec(mean(scenarios_gamma_2params, dims = 2)); rtol = 1e-3)) 
        @test(size(scenarios_gamma_2params) == (steps_ahead, num_scenarious))

        @test(isapprox(forecast_gamma_2params["intervals"]["80"]["lower"], [quantile(scenarios_gamma_2params[t,:], 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_2params["intervals"]["80"]["upper"], [quantile(scenarios_gamma_2params[t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_2params["intervals"]["95"]["lower"], [quantile(scenarios_gamma_2params[t,:], 0.05/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_2params["intervals"]["95"]["upper"], [quantile(scenarios_gamma_2params[t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

        @test(isapprox(forecast_gamma_X["mean"], vec(mean(scenarios_gamma_X, dims = 2)); rtol = 1e-3)) 
        @test(size(scenarios_gamma_X) == (steps_ahead, num_scenarious))

        @test(isapprox(forecast_gamma_X["intervals"]["80"]["lower"], [quantile(scenarios_gamma_X[t,:], 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_X["intervals"]["80"]["upper"], [quantile(scenarios_gamma_X[t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_X["intervals"]["95"]["lower"], [quantile(scenarios_gamma_X[t,:], 0.05/2) for t in 1:steps_ahead]))
        @test(isapprox(forecast_gamma_X["intervals"]["95"]["upper"], [quantile(scenarios_gamma_X[t,:], 1 - 0.05/2) for t in 1:steps_ahead]))
    end

    #@info(" --- Test quality of fit and forecast - Gamma")
    @testset "quality of fit - Gamma" begin
        y         = time_series[1:end-steps_ahead,5]
        y_test    = time_series[end-steps_ahead+1:end, 5]
        
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.GammaDistribution(), [true, false],
                                                        1.0, "random walk slope", "deterministic 12", missing)
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

        @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e0))
        @test(isapprox(forec["mean"], y_test; rtol = 1e0))
    end

    #@info(" --- Test quality of fit - Gamma with 2 params")
    # @testset "quality of fit - Gamma with 2 params" begin
    #     y         = time_series[1:end-steps_ahead,5]
    #     y_test    = time_series[end-steps_ahead+1:end, 5]
    #     gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.GammaDistribution(), [true, true],
    #                                                     0.0, ["random walk slope", "random walk"], ["deterministic 12", "deterministic 12"], [missing, missing])
    #     fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    #     forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)
            
    #     @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e0))
    #     @test(isapprox(forec["mean"], y_test; rtol = 1e0))
    # end

    # #@info(" --- Test quality of fit - Gamma with robust")
    # @testset "quality of fit - Gamma with robust" begin
    #     y         = time_series[1:end-steps_ahead,5]
    #     y_test    = time_series[end-steps_ahead+1:end, 5]
    #     gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.GammaDistribution(), [true, false],
    #                                                     1.0, "random walk slope", "deterministic 12", 1)
    #     fitted_model = UnobservedComponentsGAS.fit(gas_model, y; α = 0.0, robust = true)
    #     forec        = UnobservedComponentsGAS.predict(gas_model, fitted_model, y, steps_ahead, num_scenarious)

    #     @test(isapprox(fitted_model.fit_in_sample[2:end], y[2:end]; rtol = 1e0))
    #     @test(isapprox(forec["mean"], y_test; rtol = 1e0))
    # end

    @testset "AR(1) level" begin
        y         = time_series[1:end-steps_ahead,5]
        y_test    = time_series[end-steps_ahead+1:end, 5]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.GammaDistribution(), [true, false],
                                                        0.5, ["ar(1)", ""], ["deterministic 12", ""], [missing, missing])
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
        
        @test(all(fitted_model.components["param_1"]["level"]["value"] .!= zeros(length(y))))
    end
    

end