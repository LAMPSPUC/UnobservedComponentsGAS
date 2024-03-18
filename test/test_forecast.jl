@testset "Forecast" begin
    
    """
    Pensar ainda como testar a qualidade da previsão.
    Por enquanto, coloquei apenas testes simples de estrutura do output da previsão
    """

    steps_ahead    = 24
    num_scenarious = 500


    # path_dir = "test\\"
    path_dir = @__DIR__
    
    time_series_normal    = CSV.read(joinpath(path_dir, "data/timeseries_normal_rws_d1.csv"), DataFrame)
    time_series_lognormal = CSV.read(joinpath(path_dir, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    time_series_t         = CSV.read(joinpath(path_dir, "data/timeseries_t_rws_d1.csv"), DataFrame)

    @info(" ---  ---------- Test for normal distribution ---------- ")
    y_normal         = time_series_normal[:,1]
    T                = length(y_normal)
    X_normal         = hcat(y_normal.+5*rand(T), y_normal.+10*rand(T))
    X_normal_forec   = hcat(y_normal[end-steps_ahead+1:end].+5*rand(steps_ahead), y_normal[end-steps_ahead+1:end].+10*rand(steps_ahead))
    dist_normal      = UnobservedComponentsGAS.NormalDistribution()
    gas_model_normal = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, true)
    gas_model_normal_X = deepcopy(gas_model_normal)    

    fitted_model_normal   = UnobservedComponentsGAS.fit(gas_model_normal, y_normal)
    fitted_model_normal_X = UnobservedComponentsGAS.fit(gas_model_normal, y_normal, X_normal)
    forecast_normal       = UnobservedComponentsGAS.predict(gas_model_normal, fitted_model_normal, y_normal, steps_ahead, num_scenarious)
    forecast_normal_X     = UnobservedComponentsGAS.predict(gas_model_normal_X, fitted_model_normal_X, y_normal, X_normal_forec, steps_ahead, num_scenarious)

    @test(isapprox(forecast_normal["mean"], vec(mean(forecast_normal["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal["intervals"]["80"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["80"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @test(isapprox(forecast_normal_X["mean"], vec(mean(forecast_normal_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal_X["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal_X["intervals"]["80"]["lower"], [quantile(forecast_normal_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["80"]["upper"], [quantile(forecast_normal_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["95"]["lower"], [quantile(forecast_normal_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal_X["intervals"]["95"]["upper"], [quantile(forecast_normal_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))


    @info(" ---  ---------- Test for LogNormal distribution ---------- ")
    y_lognormal         = time_series_lognormal[:,1]
    T                   = length(y_lognormal)
    X_lognormal         = hcat(y_lognormal.+5*rand(T), y_lognormal.+10*rand(T))
    X_lognormal_forec   = hcat(y_lognormal[end-steps_ahead+1:end].+5*rand(steps_ahead), y_lognormal[end-steps_ahead+1:end].+10*rand(steps_ahead))
    dist_lognormal      = UnobservedComponentsGAS.LogNormalDistribution()
    gas_model_lognormal = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, true)

    fitted_model_lognormal   = UnobservedComponentsGAS.fit(gas_model_lognormal, y_lognormal)  
    fitted_model_lognormal_X = UnobservedComponentsGAS.fit(gas_model_lognormal, y_lognormal, X_lognormal)
    forecast_lognormal       = UnobservedComponentsGAS.predict(gas_model_lognormal, fitted_model_lognormal, y_lognormal, steps_ahead, num_scenarious)
    forecast_lognormal_X     = UnobservedComponentsGAS.predict(gas_model_lognormal, fitted_model_lognormal_X, y_lognormal, X_lognormal_forec, steps_ahead, num_scenarious)

    @test(isapprox(forecast_lognormal["mean"], vec(mean(forecast_lognormal["scenarios"], dims = 2)); rtol = 1e-1)) 
    @test(size(forecast_lognormal["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_lognormal["intervals"]["80"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    @test(isapprox(forecast_lognormal["intervals"]["80"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    @test(isapprox(forecast_lognormal["intervals"]["95"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    @test(isapprox(forecast_lognormal["intervals"]["95"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!

    # @test(isapprox(forecast_lognormal_X["mean"], vec(mean(forecast_lognormal_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_lognormal_X["scenarios"]) == (steps_ahead, num_scenarious))

    # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["80"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["lower"], [quantile(forecast_lognormal_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    # @test(isapprox(forecast_lognormal_X["intervals"]["95"]["upper"], [quantile(forecast_lognormal_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @info(" ---  ---------- Test for tLocationScaleDistribution distribution ---------- ")
    y_t         = time_series_t[:,1]
    T           = length(y_t)
    X_t         = hcat(y_t.+5*rand(T), y_t.+10*rand(T))
    X_t_forec   = hcat(y_t[end-steps_ahead+1:end].+5*rand(steps_ahead), y_t[end-steps_ahead+1:end].+10*rand(steps_ahead))
    dist_t      = UnobservedComponentsGAS.tLocationScaleDistribution()
    gas_model_t = UnobservedComponentsGAS.GASModel(dist_t, [true, false, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, true)

    fitted_model_t   = UnobservedComponentsGAS.fit(gas_model_t, y_t)
    fitted_model_t_X = UnobservedComponentsGAS.fit(gas_model_t, y_t, X_t)
    forecast_t       = UnobservedComponentsGAS.predict(gas_model_t, fitted_model_t, y_t, steps_ahead, num_scenarious)
    forecast_t_X     = UnobservedComponentsGAS.predict(gas_model_t, fitted_model_t_X, y_t, X_t_forec, steps_ahead, num_scenarious)

    @test(isapprox(forecast_t["mean"], vec(mean(forecast_t["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t["scenarios"]) == (steps_ahead, num_scenarious))
    
    @test(isapprox(forecast_t["intervals"]["80"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["80"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["95"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["95"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))
    
    @test(isapprox(forecast_t_X["mean"], vec(mean(forecast_t_X["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t_X["scenarios"]) == (steps_ahead, num_scenarious))
    
    @test(isapprox(forecast_t_X["intervals"]["80"]["lower"], [quantile(forecast_t_X["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t_X["intervals"]["80"]["upper"], [quantile(forecast_t_X["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t_X["intervals"]["95"]["lower"], [quantile(forecast_t_X["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t_X["intervals"]["95"]["upper"], [quantile(forecast_t_X["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))

end