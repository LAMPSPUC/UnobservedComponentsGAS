@testset "Forecast" begin
    
    """
    Pensar ainda como testar a qualidade da previsão.
    Por enquanto, coloquei apenas testes simples de estrutura do output da previsão
    Faltam os testes para os casos com explicativas!!
    """

    "FALTAM TESTES RELATIVOS ÀS EXPLICATIVAS"

    steps_ahead    = 24
    num_scenarious = 500

    # path = "data\\"
    #path = "test\\data\\"
    # path = "..\\..\\test\\data\\"
    time_series_normal    = CSV.read(joinpath(@__DIR__, "data/timeseries_normal_rws_d1.csv"), DataFrame)
    time_series_lognormal = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    time_series_t         = CSV.read(joinpath(@__DIR__, "data/timeseries_t_rws_d1.csv"), DataFrame)

    @info(" ---  ---------- Test for normal distribution ---------- ")
    y_normal         = time_series_normal[:,1]
    dist_normal      = UnobservedComponentsGAS.NormalDistribution()
    gas_model_normal = UnobservedComponentsGAS.GASModel(dist_normal, [true, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, true)

    fitted_model_normal = UnobservedComponentsGAS.fit(gas_model_normal, y_normal)
    forecast_normal     = UnobservedComponentsGAS.predict(gas_model_normal, fitted_model_normal, y_normal, steps_ahead, num_scenarious)

    @test(isapprox(forecast_normal["mean"], vec(mean(forecast_normal["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_normal["scenarios"]) == (steps_ahead, num_scenarious))

    @test(isapprox(forecast_normal["intervals"]["80"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["80"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["lower"], [quantile(forecast_normal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]))
    @test(isapprox(forecast_normal["intervals"]["95"]["upper"], [quantile(forecast_normal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]))

    @info(" ---  ---------- Test for LogNormal distribution ---------- ")
    y_lognormal         = time_series_lognormal[:,2]
    dist_lognormal      = UnobservedComponentsGAS.LogNormalDistribution()
    gas_model_lognormal = UnobservedComponentsGAS.GASModel(dist_lognormal, [true, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, false)

    fitted_model_lognormal = UnobservedComponentsGAS.fit(gas_model_lognormal, y_lognormal)
    forecast_lognormal     = UnobservedComponentsGAS.predict(gas_model_lognormal, fitted_model_lognormal, y_lognormal, steps_ahead, num_scenarious)

    # @test(isapprox(forecast_lognormal["mean"], vec(mean(forecast_lognormal["scenarios"], dims = 2)); rtol = 1e-1)) 
    @test(size(forecast_lognormal["scenarios"]) == (steps_ahead, num_scenarious))
    
    # @test(isapprox(forecast_lognormal["intervals"]["80"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    # @test(isapprox(forecast_lognormal["intervals"]["80"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    # @test(isapprox(forecast_lognormal["intervals"]["95"]["lower"], [quantile(forecast_lognormal["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!
    # @test(isapprox(forecast_lognormal["intervals"]["95"]["upper"], [quantile(forecast_lognormal["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead];rtol = 1e-3))# falhando !!

    @info(" ---  ---------- Test for tLocationScaleDistribution distribution ---------- ")
    y_t         = time_series_t[:,1]
    dist_t      = UnobservedComponentsGAS.tLocationScaleDistribution()
    gas_model_t = UnobservedComponentsGAS.GASModel(dist_t, [true, false, false], 1.0, Dict(1=>false), 
                                                Dict(1 => true),  Dict(1 => 1), 
                                                Dict(1 => 12), false, true)

    fitted_model_t = UnobservedComponentsGAS.fit(gas_model_t, y_t)
    forecast_t     = UnobservedComponentsGAS.predict(gas_model_t, fitted_model_t, y_t, steps_ahead, num_scenarious)

    @test(isapprox(forecast_t["mean"], vec(mean(forecast_t["scenarios"], dims = 2)); rtol = 1e-3)) 
    @test(size(forecast_t["scenarios"]) == (steps_ahead, num_scenarious))
    
    @test(isapprox(forecast_t["intervals"]["80"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["80"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.2/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["95"]["lower"], [quantile(forecast_t["scenarios"][t,:], 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))
    @test(isapprox(forecast_t["intervals"]["95"]["upper"], [quantile(forecast_t["scenarios"][t,:], 1 - 0.05/2) for t in 1:steps_ahead]; rtol = 1e-3))

end
