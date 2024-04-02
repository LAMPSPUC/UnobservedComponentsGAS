@testset "Update Model" begin
    
    function compare_dictionaries(dict1::Dict, dict2::Dict; key_prefix::String="", rtol::Float64=1e-3)
        # Check if dictionaries have the same keys
        keys1 = keys(dict1)
        keys2 = keys(dict2)
        if keys1 != keys2
            # println("Error: Dictionaries have different keys.")
            return false
        end
        
        # Compare each (key, value) pair
        for key in keys1
            value1 = dict1[key]
            value2 = dict2[key]
            if isa(value1, Dict) && isa(value2, Dict)
                # Recursively compare nested dictionaries
                if !compare_dictionaries(value1, value2, "$key_prefix$key.")
                    return false
                end
            elseif !isapprox(value1, value2; rtol=rtol)
                println("Difference found at key: $key_prefix$key")
                println("Value in dict1: $value1")
                println("Value in dict2: $value2")
                return false
            end
        end
        
        return true
    end    

    # path = "data\\"
    # path = "test\\data\\"
    #path = "..\\..\\test\\data\\"
    time_series_normal    = CSV.read(joinpath(@__DIR__, "data/timeseries_normal_rws_d1.csv"), DataFrame)
    time_series_lognormal = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    time_series_t         = CSV.read(joinpath(@__DIR__, "data/timeseries_t_rws_d1.csv"), DataFrame)

    @info(" --- Test for normal distribution")
    y_normal         = time_series_normal[:,1]
    dist_normal      = UnobservedComponentsGAS.NormalDistribution()
    gas_model_normal = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 0.0, "random walk slope", "deterministic 12", 1)
    # gas_model_normal = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 0.0, Dict(1=>false), 
    #                                                     Dict(1 => true),  Dict(1 => false), 
    #                                                     Dict(1 => 12), false, false)

    fitted_model_normal = UnobservedComponentsGAS.fit(gas_model_normal, y_normal)
    # Me parece que essa função só funciona para a sazo estocástica -> ao mesmo tempo, a definição da sazo na linha 124 está determinística
    updated_model_normal = UnobservedComponentsGAS.update_fitted_params_and_components_dict(gas_model_normal, fitted_model_normal, y_normal, missing)

    #Trocar tudo por isapprox!!!

    @test(all(updated_model_normal.fit_in_sample .==  fitted_model_normal.fit_in_sample))
    @test(isapprox(updated_model_normal.fitted_params["param_1"], fitted_model_normal.fitted_params["param_1"]; rtol=1e-3))
    @test(isapprox(updated_model_normal.fitted_params["param_2"], fitted_model_normal.fitted_params["param_2"]; rtol = 1e-3))
    @test(compare_dictionaries(fitted_model_normal.residuals, updated_model_normal.residuals))

    @test(compare_dictionaries(fitted_model_normal.components["param_1"]["slope"]["hyperparameters"], updated_model_normal.components["param_1"]["slope"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_normal.components["param_1"]["level"]["hyperparameters"], updated_model_normal.components["param_1"]["level"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_normal.components["param_1"]["seasonality"]["hyperparameters"], updated_model_normal.components["param_1"]["seasonality"]["hyperparameters"])) 
    
    # @test(compare_dictionaries(fitted_model_normal.components["param_1"]["slope"], updated_model_normal.components["param_1"]["slope"])) 
    # @test(compare_dictionaries(fitted_model_normal.components["param_1"]["level"], updated_model_normal.components["param_1"]["level"])) 
    # @test(compare_dictionaries(fitted_model_normal.components["param_1"]["seasonality"], updated_model_normal.components["param_1"]["seasonality"])) 

    @test(isequal(fitted_model_normal.components["param_1"]["intercept"], updated_model_normal.components["param_1"]["intercept"])) 
    @test(isequal(fitted_model_normal.information_criteria, updated_model_normal.information_criteria)) 


    @info(" --- Test for LogNormal distribution")
    y_lognormal         = time_series_lognormal[:,1]
    dist_lognormal      = UnobservedComponentsGAS.LogNormalDistribution()
    gas_model_lognormal = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false], 0.0, Dict(1=>false), 
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, false)

    fitted_model_lognormal = UnobservedComponentsGAS.fit(gas_model_lognormal, y_lognormal)
    updated_model_lognormal = UnobservedComponentsGAS.update_fitted_params_and_components_dict(gas_model_lognormal, fitted_model_lognormal, y_lognormal, missing)

    @test(all(updated_model_lognormal.fit_in_sample .==  fitted_model_lognormal.fit_in_sample))
    # @test(isapprox(updated_model_lognormal.fitted_params["param_1"], fitted_model_lognormal.fitted_params["param_1"]; rtol=1e-2))
    # @test(isapprox(updated_model_lognormal.fitted_params["param_2"], fitted_model_lognormal.fitted_params["param_2"]; rtol=1e-2))
    # @test(compare_dictionaries(fitted_model_lognormal.residuals, updated_model_lognormal.residuals))

    @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["slope"]["hyperparameters"], updated_model_lognormal.components["param_1"]["slope"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["level"]["hyperparameters"], updated_model_lognormal.components["param_1"]["level"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["seasonality"]["hyperparameters"], updated_model_lognormal.components["param_1"]["seasonality"]["hyperparameters"])) 

    # @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["slope"], updated_model_lognormal.components["param_1"]["slope"])) # κ é igual, mas componente não
    # @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["level"], updated_model_lognormal.components["param_1"]["level"])) # κ é igual, mas componente não
    # @test(compare_dictionaries(fitted_model_lognormal.components["param_1"]["seasonality"], updated_model_lognormal.components["param_1"]["seasonality"])) # κ é igual, mas componente não

    @test(isequal(fitted_model_lognormal.components["param_1"]["intercept"], updated_model_lognormal.components["param_1"]["intercept"])) 
    @test(isequal(fitted_model_lognormal.information_criteria, updated_model_lognormal.information_criteria)) 

    
    @info(" --- Test for tLocationScaleDistribution distribution")
    y_t         = time_series_t[:,1]
    dist_t      = UnobservedComponentsGAS.tLocationScaleDistribution()
    gas_model_t = UnobservedComponentsGAS.GASModel(dist_t, [true, false, false], 0.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => false), 
                                                Dict(1 => 12), false, false)

    fitted_model_t = UnobservedComponentsGAS.fit(gas_model_t, y_t)
    # Me parece que essa função só funciona para a sazo estocástica -> ao mesmo tempo, a definição da sazo na linha 124 está determinística
    updated_model_t = UnobservedComponentsGAS.update_fitted_params_and_components_dict(gas_model_t, fitted_model_t, y_t, missing)

    @test(all(updated_model_t.fit_in_sample .==  fitted_model_t.fit_in_sample))
    @test(isapprox(updated_model_t.fitted_params["param_1"], fitted_model_t.fitted_params["param_1"]; rtol=1e-3))
    @test(isapprox(updated_model_t.fitted_params["param_2"], fitted_model_t.fitted_params["param_2"]; rtol=1e-3))
    @test(compare_dictionaries(fitted_model_t.residuals, updated_model_t.residuals))
    
    @test(compare_dictionaries(fitted_model_t.components["param_1"]["slope"]["hyperparameters"], updated_model_t.components["param_1"]["slope"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_t.components["param_1"]["level"]["hyperparameters"], updated_model_t.components["param_1"]["level"]["hyperparameters"]))
    @test(compare_dictionaries(fitted_model_t.components["param_1"]["seasonality"]["hyperparameters"], updated_model_t.components["param_1"]["seasonality"]["hyperparameters"])) 
    
    # @test(compare_dictionaries(fitted_model_t.components["param_1"]["slope"], updated_model_t.components["param_1"]["slope"])) 
    # @test(compare_dictionaries(fitted_model_t.components["param_1"]["level"], updated_model_t.components["param_1"]["level"])) 
    # @test(compare_dictionaries(fitted_model_t.components["param_1"]["seasonality"], updated_model_t.components["param_1"]["seasonality"])) 

    @test(isequal(fitted_model_t.components["param_1"]["intercept"], updated_model_t.components["param_1"]["intercept"])) 
    @test(isequal(fitted_model_t.information_criteria, updated_model_t.information_criteria)) 
end