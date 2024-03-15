@testset "Initialization" begin
    
    # path = "data\\"
    # path = "test\\data\\"
    # time_series = CSV.read(path*"timeseries_normal_rws_d1.csv", DataFrame)

    T = 100
    Random.seed!(123)
    y = rand(T)

    stochastic       = false
    order            = [1]
    max_order        = 1
    X                = missing

    @info("Test with Random Walk and Slope")
    has_level        = true
    has_slope        = true
    has_seasonality  = true
    seasonal_period = 12

    if has_level && !has_slope
        trend = "local level"
    elseif has_level && has_slope
        trend = "local linear trend"
    end

    if has_seasonality && stochastic
        seasonal = "stochastic "*string(seasonal_period)
    elseif has_seasonality && !stochastic
        seasonal = "deterministic "*string(seasonal_period)
    end
        
    state_space_model = UnobservedComponents(y ; trend = trend ,seasonal = seasonal) 
    StateSpaceModels.fit!(state_space_model)
    pred_state      = StateSpaceModels.get_predictive_state(state_space_model)
    initial_level   = pred_state[2:end,1]
    initial_slope   = pred_state[2:end,2]
    initial_seasonality = zeros(length(y))
    for t in  1:length(y)
        initial_seasonality[t] = -sum(pred_state[t+1, end-(seasonal_period-2):end])
    end
    
    initial_values    = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rws"]["values"], initial_level; rtol = 1e-3))
    @test(isapprox(initial_values["slope"]["values"], initial_slope; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_seasonality; rtol = 1e-3))
    @test(all(initial_values["ar"]["values"] .!= zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))

    @info("Test with Random Walk")
    has_level        = true
    has_slope        = false
    has_seasonality  = true
    seasonal_period = 12

    if has_level && !has_slope
        trend = "local level"
    elseif has_level && has_slope
        trend = "local linear trend"
    end

    if has_seasonality && stochastic
        seasonal = "stochastic "*string(seasonal_period)
    elseif has_seasonality && !stochastic
        seasonal = "deterministic "*string(seasonal_period)
    end
        
    state_space_model = UnobservedComponents(y ; trend = trend ,seasonal = seasonal) 
    StateSpaceModels.fit!(state_space_model)
    pred_state = StateSpaceModels.get_predictive_state(state_space_model)

    state_space_model = UnobservedComponents(y ; trend = trend ,seasonal = seasonal) 
    StateSpaceModels.fit!(state_space_model)
    pred_state      = StateSpaceModels.get_predictive_state(state_space_model)
    initial_level   = pred_state[2:end,1]
    initial_seasonality = zeros(length(y))
    for t in  1:length(y)
        initial_seasonality[t] = -sum(pred_state[t+1, end-(seasonal_period-2):end])
    end

    initial_values    = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rw"]["values"], initial_level; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_seasonality; rtol = 1e-3))
    @test(all(initial_values["ar"]["values"] .!= zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))

    @info("Test with Random Walk and two time varying parameters")
    has_level        = true
    has_slope        = false
    has_seasonality  = true
    seasonal_period = 12

    if has_level && !has_slope
        trend = "local level"
    elseif has_level && has_slope
        trend = "local linear trend"
    end

    if has_seasonality && stochastic
        seasonal = "stochastic "*string(seasonal_period)
    elseif has_seasonality && !stochastic
        seasonal = "deterministic "*string(seasonal_period)
    end
        
    state_space_model = UnobservedComponents(y ; trend = trend ,seasonal = seasonal) 
    StateSpaceModels.fit!(state_space_model)
    pred_state = StateSpaceModels.get_predictive_state(state_space_model)

    state_space_model = UnobservedComponents(y ; trend = trend ,seasonal = seasonal) 
    StateSpaceModels.fit!(state_space_model)
    pred_state      = StateSpaceModels.get_predictive_state(state_space_model)
    initial_level   = pred_state[2:end,1]
    initial_seasonality = zeros(length(y))
    for t in  1:length(y)
        initial_seasonality[t] = -sum(pred_state[t+1, end-(seasonal_period-2):end])
    end

    Random.seed!(125)
    y = rand(T) .+ collect(1.:T) .+ sin.(collect(1.:T))
    gas_model = UnobservedComponentsGAS.GASModel(UnobservedComponentsGAS.NormalDistribution(), [true, true], 0.0, Dict(1=>has_level, 2=>has_level), 
                                                Dict(1 => has_slope, 2 => has_slope),  Dict(1 => false, 2 => false), 
                                                Dict(1 => seasonal_period, 2=> seasonal_period), false, false)
    initial_values = UnobservedComponentsGAS.create_output_initialization(y, X, gas_model)

    @test(isapprox(initial_values["rw"]["values"], initial_level; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_seasonality; rtol = 1e-3))
    @test(all(initial_values["ar"]["values"] .!= zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))


    @info("Test create_output_initialization_from_fit")
    dist = UnobservedComponentsGAS.NormalDistribution(missing, missing)
    gas_model = UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, Dict(1=>false, 2=>false),  
                                            Dict(1 => true, 2=>false),  Dict(1 => 1), 
                                            Dict(1 => 12, 2 => 12), false, false)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    output_initial_values = UnobservedComponentsGAS.create_output_initialization_from_fit(fitted_model, gas_model)
    
    @test(all(output_initial_values["rw"]["values"] .== 0))
    @test(all(output_initial_values["seasonality"]["values"] .!= 0))
    @test(all(output_initial_values["ar"]["values"] .!= 0))
    @test(all(output_initial_values["slope"]["values"] .!= zeros(T)))
    @test(all(output_initial_values["rws"]["values"] .!=  zeros(T)))

end