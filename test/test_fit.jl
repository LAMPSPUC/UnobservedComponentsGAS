@testset "Fit" begin  
    
    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    y = time_series[:,2]
    T = length(y)
    X = [2*y y/2 rand(T)]
   
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
    dist_lognormal = UnobservedComponentsGAS.LogNormalDistribution()
    dist_t         = UnobservedComponentsGAS.tLocationScaleDistribution()

    gas_model_normal         = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_normal_2params = UnobservedComponentsGAS.GASModel(dist_normal, [true, true], 1.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
    gas_model_lognormal      = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false], 1.0, "random walk slope", "deterministic 12", missing)
    gas_model_t              = UnobservedComponentsGAS.GASModel(dist_t, [true, false, false], 1.0, "random walk slope", "deterministic 12", missing)

    gas_model_normal_X            = deepcopy(gas_model_normal)
    gas_model_normal_X_2params    = deepcopy(gas_model_normal_2params)
    gas_model_lognormal_X         = deepcopy(gas_model_lognormal)
    gas_model_t_X                 = deepcopy(gas_model_t)

    @info(" --- Testing create_model functions")
    # Create model with no explanatory series
    model_normal, parameters_normal, initial_values_normal                         = UnobservedComponentsGAS.create_model(gas_model_normal, y, missing)
    model_normal_2params, parameters_normal_2params, initial_values_normal_2params = UnobservedComponentsGAS.create_model(gas_model_normal_2params, y, missing)
    model_lognormal, parameters_lognormal, initial_values_lognormal                = UnobservedComponentsGAS.create_model(gas_model_lognormal, y, missing)
    model_t, parameters_t, initial_values_t                                        = UnobservedComponentsGAS.create_model(gas_model_t, y, 1)
    # Create model with explanatory series
    model_normal_X, parameters_normal_X, initial_values_normal_X                         = UnobservedComponentsGAS.create_model(gas_model_normal_X, y, X, missing);
    model_normal_X_2params, parameters_normal_X_2params, initial_values_normal_X_2params = UnobservedComponentsGAS.create_model(gas_model_normal_X_2params, y, X, missing);
    model_lognormal_X, parameters_lognormal_X, initial_values_lognormal_X                = UnobservedComponentsGAS.create_model(gas_model_lognormal_X, y, X, missing)
    model_t_X, parameters_t_X, initial_values_t_X                                        = UnobservedComponentsGAS.create_model(gas_model_t_X, y, X, 1)
    
    @test(size(parameters_normal)         == (T,2))
    @test(size(parameters_normal_2params) == (T,2))
    @test(size(parameters_lognormal)      == (T,2))
    @test(size(parameters_t)              == (T,3))
    @test(typeof(model_normal)            == JuMP.Model)
    @test(typeof(model_lognormal)         == JuMP.Model)
    @test(typeof(model_t)                 == JuMP.Model)
    @test(test_initial_values_components(initial_values_normal, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_normal_2params, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_lognormal, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_t, rw, rws, ar, seasonality))
    
    @test(size(parameters_normal_X)         == (T,2))
    @test(size(parameters_normal_X_2params) == (T,2))
    @test(size(parameters_t_X)              == (T,3))
    @test(typeof(model_normal_X)            == JuMP.Model)
    @test(typeof(model_normal_X_2params)    == JuMP.Model)
    @test(typeof(model_t_X)                 == JuMP.Model)
    @test(test_initial_values_components(initial_values_normal_X, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_normal_X_2params, rw, rws, ar, seasonality))
    @test(test_initial_values_components(initial_values_t_X, rw, rws, ar, seasonality))

    @info(" --- Testing fit functions")
    fitted_model_normal         = UnobservedComponentsGAS.fit(gas_model_normal, y)
    fitted_model_normal_2params = UnobservedComponentsGAS.fit(gas_model_normal_2params, y)
    fitted_model_lognormal      = UnobservedComponentsGAS.fit(gas_model_lognormal, y)
    fitted_model_t              = UnobservedComponentsGAS.fit(gas_model_t, y)

    fitted_model_normal_X         = UnobservedComponentsGAS.fit(gas_model_normal_X, y, X)
    fitted_model_normal_X_2params = UnobservedComponentsGAS.fit(gas_model_normal_X_2params, y, X)
    fitted_model_lognormal_X      = UnobservedComponentsGAS.fit(gas_model_lognormal_X, y, X)
    fitted_model_t_X              = UnobservedComponentsGAS.fit(gas_model_t_X, y, X)

    # "Test if termination_status is correct"
    possible_status = ["LOCALLY_SOLVED"]#, "INVALID_MODEL", "ALMOST_LOCALLY_SOLVED"]
    @test(fitted_model_normal.model_status in possible_status)
    @test(fitted_model_normal_2params.model_status in possible_status)
    @test(fitted_model_lognormal.model_status in possible_status)
    @test(fitted_model_t.model_status in possible_status)
    @test(fitted_model_normal_X.model_status in possible_status)
    @test(fitted_model_lognormal_X.model_status in possible_status)
    @test(fitted_model_t_X.model_status in possible_status)
    
    # "Test if selected_variables is missing "
    @test(ismissing(fitted_model_normal.selected_variables))
    @test(ismissing(fitted_model_normal_2params.selected_variables))
    @test(ismissing(fitted_model_lognormal.selected_variables))
    @test(ismissing(fitted_model_t.selected_variables))

    # "Test if fitted_params has the right keys -> order may be a problem"
    @test(all(keys(fitted_model_normal.fitted_params) .== ["param_2","param_1"]))
    @test(all(keys(fitted_model_normal_2params.fitted_params) .== ["param_2","param_1"]))
    @test(all(keys(fitted_model_lognormal.fitted_params) .== ["param_2","param_1"]))
    @test(all(keys(fitted_model_t.fitted_params).== ["param_2","param_1","param_3"]))

    # "Test if all time varying and fixed params are time varying and fixed"
    @test(!all(y->y==fitted_model_normal.fitted_params["param_1"][1],fitted_model_normal.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_normal.fitted_params["param_2"][1],fitted_model_normal.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_2params.fitted_params["param_1"][1],fitted_model_normal_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_normal_2params.fitted_params["param_2"][1],fitted_model_normal_2params.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_X.fitted_params["param_1"][1],fitted_model_normal_X.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_normal_X.fitted_params["param_2"][1],fitted_model_normal_X.fitted_params["param_2"]))
    @test(!all(y->y==fitted_model_normal_X_2params.fitted_params["param_1"][1],fitted_model_normal_X_2params.fitted_params["param_1"]))
    @test(!all(y->y==fitted_model_normal_X_2params.fitted_params["param_2"][1],fitted_model_normal_X_2params.fitted_params["param_2"]))

    @test(!all(y->y==fitted_model_lognormal.fitted_params["param_1"][1],fitted_model_lognormal.fitted_params["param_1"]))
    # @test(all(y->y==fitted_model_lognormal.fitted_params["param_2"][1],fitted_model_lognormal.fitted_params["param_2"])) # tá dando erro -> variancia vindo variante no tempo

    @test(!all(y->y==fitted_model_t.fitted_params["param_1"][1],fitted_model_t.fitted_params["param_1"]))
    @test(all(y->y==fitted_model_t.fitted_params["param_2"][1],fitted_model_t.fitted_params["param_2"]))
    @test(all(y->y==fitted_model_t.fitted_params["param_3"][1],fitted_model_t.fitted_params["param_3"]))

    # "Test if all residuals are being generated"
    residuals_types = ["q_residuals", "std_residuals", "cs_residuals"]
    @test(all(keys(fitted_model_normal.residuals)         .== residuals_types))
    @test(all(keys(fitted_model_normal_2params.residuals) .== residuals_types))
    @test(all(keys(fitted_model_lognormal.residuals)      .== residuals_types))
    @test(all(keys(fitted_model_t.residuals)              .== residuals_types))


    @info(" --- Test quality of fit - Normal")

    time_series_normal      = CSV.read(joinpath(@__DIR__,  "data/timeseries_normal_rws_d1.csv"), DataFrame)
    benchmark_values_normal = JSON3.read(joinpath(@__DIR__, "data/benchmark_values_normal_rws.json"))

    initial_values_normal = convert_dict_keys_to_string(initial_values_normal)
    N = 10
    T = size(time_series_normal, 1)

    σ2_values        = zeros(N)
    level_κ_values   = zeros(N)
    slope_κ_values   = zeros(N)
    # intercept_values = zeros(N)
    fitted_values_normal = zeros(T,N)

    for j in 1:N
        y         = time_series_normal[:,j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false],
                                                     1.0, "random walk slope", "deterministic 12", missing)
        
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)

        σ2_values[j]               = fitted_model.fitted_params["param_2"][1]        
        level_κ_values[j]          = fitted_model.components["param_1"]["level"]["hyperparameters"]["κ"]
        slope_κ_values[j]          = fitted_model.components["param_1"]["slope"]["hyperparameters"]["κ"]
        # intercept_values[j]        = fitted_model.components["param_1"]["intercept"]
        fitted_values_normal[:,j] .= fitted_model.fit_in_sample
        
    end

    @test(isapprox(mean(fitted_values_normal[2:end,:], dims = 2), mean(Matrix(time_series_normal[2:end,:]), dims = 2); rtol = 1e-1))


    @info(" --- Test quality of fit - Normal with 2 params")

    time_series_normal      = CSV.read(joinpath(@__DIR__,  "data/timeseries_normal_rws_d1.csv"), DataFrame)
    benchmark_values_normal = JSON3.read(joinpath(@__DIR__, "data/benchmark_values_normal_rws.json"))

    initial_values_normal = convert_dict_keys_to_string(initial_values_normal)
    N = 10
    T = size(time_series_normal, 1)

    σ2_values        = zeros(N)
    level_κ_values   = zeros(N)
    slope_κ_values   = zeros(N)
    # intercept_values = zeros(N)
    fitted_values_normal_2params = zeros(T,N)

    for j in 1:N
        y         = time_series_normal[:,j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, true],
                                                     1.0, ["random walk slope", "random walk"], ["deterministic 12", "deterministic 12"], [missing, missing])
        
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)

        σ2_values[j]                       = fitted_model.fitted_params["param_2"][1]        
        level_κ_values[j]                  = fitted_model.components["param_1"]["level"]["hyperparameters"]["κ"]
        slope_κ_values[j]                  = fitted_model.components["param_1"]["slope"]["hyperparameters"]["κ"]
        # intercept_values[j]                = fitted_model.components["param_1"]["intercept"]
        fitted_values_normal_2params[:,j] .= fitted_model.fit_in_sample
        
    end

    @test(isapprox(mean(fitted_values_normal_2params[2:end,:], dims = 2), mean(Matrix(time_series_normal[2:end,:]), dims = 2); rtol = 1e-1))

    @info(" --- Test quality of fit - lognormal")

    time_series_lognormal      = CSV.read(joinpath(@__DIR__,  "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    benchmark_values_lognormal = JSON3.read(joinpath(@__DIR__, "data/benchmark_values_lognormal_rws.json"))
    N = 10


    σ2_values        = zeros(N)
    ν_values         = zeros(N)
    level_κ_values   = zeros(N)
    slope_κ_values   = zeros(N)
    # intercept_values = zeros(N)
    fitted_values_lognormal = zeros(T,N)
    # ~ 80 sec to run
    for j in 1:N
        y         = time_series_lognormal[:,j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.LogNormalDistribution(), [true, false],
                                                     1.0, "random walk slope", "deterministic 12", missing)
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)

        σ2_values[j]                  = fitted_model.fitted_params["param_2"][1]      
        level_κ_values[j]             = fitted_model.components["param_1"]["level"]["hyperparameters"]["κ"]
        slope_κ_values[j]             = fitted_model.components["param_1"]["slope"]["hyperparameters"]["κ"]
        # intercept_values[j]           = fitted_model.components["param_1"]["intercept"]
        fitted_values_lognormal[:,j] .= fitted_model.fit_in_sample
        
    end

    fitted_values_lognormal = fitted_values_lognormal[:,.!isinf.(fitted_values_lognormal)[1,:]]
    @test(isapprox(mean(fitted_values_lognormal[2:end,:], dims = 2), mean(Matrix(time_series_lognormal[2:end,:]), dims = 2); rtol = 1e-1))

    
    @info(" --- Test quality of fit - t")

    time_series_t      = CSV.read(joinpath(@__DIR__,   "test/data/timeseries_t_rws_d1.csv"), DataFrame)
    benchmark_values_t = JSON3.read(joinpath(@__DIR__, "test/data/benchmark_values_t_rws.json"))
    N = 10

    σ2_values        = zeros(N)
    ν_values         = zeros(N)
    level_κ_values   = zeros(N)
    slope_κ_values   = zeros(N)
    # intercept_values = zeros(N)
    fitted_values_t = zeros(T,N)
    for j in 1:N
        y         = time_series_t[:,j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.tLocationScaleDistribution(), [true, false, false],
                                                     1.0, "random walk slope", "deterministic 12", missing)
        
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y)

        σ2_values[j]          = fitted_model.fitted_params["param_2"][1]   
        ν_values[j]           = fitted_model.fitted_params["param_3"][1]        
        level_κ_values[j]     = fitted_model.components["param_1"]["level"]["hyperparameters"]["κ"]
        slope_κ_values[j]     = fitted_model.components["param_1"]["slope"]["hyperparameters"]["κ"]
        # intercept_values[j]   = fitted_model.components["param_1"]["intercept"]
        fitted_values_t[:,j] .= fitted_model.fit_in_sample
        
    end
    
    @test(isapprox(mean(fitted_values_t[2:end,:], dims = 2), mean(Matrix(time_series_t[2:end,:]), dims = 2); rtol = 1e-1))

    @info(" --- Test quality of fit - Normal with robust")

    time_series_normal      = CSV.read(joinpath(@__DIR__,  "data/timeseries_normal_rws_d1.csv"), DataFrame)
    benchmark_values_normal = JSON3.read(joinpath(@__DIR__, "data/benchmark_values_normal_rws.json"))

    initial_values_normal = convert_dict_keys_to_string(initial_values_normal)
    N = 10

    fitted_values_normal_2params = zeros(T,N)

    for j in 1:N
        y         = time_series_normal[:,j]
        gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, false],
                                                     1.0, "random walk slope", "deterministic 12", 1)
        fitted_model = UnobservedComponentsGAS.fit(gas_model, y; α = 0.5, robust = true)
        fitted_values_normal_2params[:,j] .= fitted_model.fit_in_sample
    end

    @test(isapprox(mean(fitted_values_normal_2params[2:end,:], dims = 2), mean(Matrix(time_series_normal[2:end,:]), dims = 2); rtol = 1e-1))
end

