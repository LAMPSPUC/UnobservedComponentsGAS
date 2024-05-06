@testset "Initialization" begin
    
    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_normal_rws_d1.csv"), DataFrame)
    y = time_series[:,1]
    T = length(y)
    X = rand(T, 2).+10
    X_missing  = missing
    stochastic = false


    @info("Test with Random Walk and Slope without AR and without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = true
    has_seasonality  = true
    has_ar1_level    = false
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)    
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rws"]["values"], initial_values_state_space["level"]; rtol = 1e-3))
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))


    @info("Test with Random Walk with AR and without explanatory")
    order            = [1]
    max_order        = 2
    has_level        = true
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rw"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .!= zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))

    @info("Test with just Seasonality without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["rw"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))

    @info("Test with Random Walk without seasonality and without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rw"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))
    @test(all(initial_values["slope"]["values"] .==  zeros(T)))
    @test(all(initial_values["seasonality"]["values"] .==  zeros(T)))

    @info("Test with Random Walk Slope without seasonality and without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = true
    has_ar1_level    = false
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rws"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))
    @test(all(initial_values["seasonality"]["values"] .==  zeros(T)))

    @info("Test with AR(1) with seasonality and without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = true
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["ar1_level"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))


    @info("Test with Random Walk and Slope without AR and with explanatory")
    stochastic       = false
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = true
    has_ar1_level    = false
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)    
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rws"]["values"], initial_values_state_space["level"]; rtol = 1e-3))
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))

    @info("Test with Random Walk with AR and with explanatory")
    order            = [1]
    max_order        = 2
    has_level        = true
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rw"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .!= zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))

    @info("Test with just Seasonality with explanatory") 
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["rw"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))

    @info("Test with Random Walk without seasonality and without explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = false
    has_ar1_level    = false
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rw"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rws"]["values"] .==  zeros(T)))
    @test(all(initial_values["slope"]["values"] .==  zeros(T)))
    @test(all(initial_values["seasonality"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))


    @info("Test with Random Walk Slope without seasonality and without explanatory") 
    order            = [nothing]
    max_order        = 0
    has_level        = true
    has_slope        = true
    has_ar1_level    = false
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["rws"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["slope"]["values"], initial_values_state_space["slope"]; rtol = 1e-3))
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))
    @test(all(initial_values["seasonality"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))


    @info("Test with AR(1) without seasonality and without explanatory") 
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = true
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X_missing, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["ar1_level"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))

    @info("Test with AR(1) without seasonality and with explanatory") 
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = true
    has_seasonality  = false
    seasonal_period  = missing

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["ar1_level"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))
    

    @info("Test with AR(1) with seasonality and with explanatory")
    order            = [nothing]
    max_order        = 0
    has_level        = false
    has_slope        = false
    has_ar1_level    = true
    has_seasonality  = true
    seasonal_period  = 12

    initial_values_state_space = UnobservedComponentsGAS.define_state_space_model(y, X, (has_level || has_ar1_level), has_slope, has_seasonality, seasonal_period, stochastic)
    initial_values             = UnobservedComponentsGAS.get_initial_values(y, X, has_level, has_ar1_level, has_slope, has_seasonality, seasonal_period, stochastic, order, max_order)

    @test(isapprox(initial_values["ar1_level"]["values"], initial_values_state_space["level"]; rtol = 1e-3)) 
    @test(isapprox(initial_values["seasonality"]["values"],initial_values_state_space["seasonality"]; rtol = 1e-3))
    @test(all(initial_values["seasonality"]["γ"] .== initial_values_state_space["γ"]))
    @test(all(initial_values["seasonality"]["γ_star"] .== initial_values_state_space["γ_star"]))
    @test(all(initial_values["ar"]["values"] .== zeros(T)))
    @test(all(initial_values["slope"]["values"] .== zeros(T)))
    @test(all(initial_values["rw"]["values"] .==  zeros(T)))
    @test(all(initial_values["explanatories"] .== initial_values_state_space["explanatory"]))


    @info("Test create_output_initialization_from_fit")
    dist = UnobservedComponentsGAS.NormalDistribution()
    gas_model = UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, "random walk slope", "deterministic 12", 1)

    fitted_model = UnobservedComponentsGAS.fit(gas_model, y)
    output_initial_values = UnobservedComponentsGAS.create_output_initialization_from_fit(fitted_model, gas_model)
    
    @test(all(output_initial_values["rw"]["values"] .== 0))
    @test(all(output_initial_values["seasonality"]["values"] .!= 0))
    @test(all(output_initial_values["ar"]["values"] .!= 0))
    @test(all(output_initial_values["slope"]["values"] .!= zeros(T)))
    @test(all(output_initial_values["rws"]["values"] .!=  zeros(T)))

    dist                  = UnobservedComponentsGAS.NormalDistribution()
    gas_model             = UnobservedComponentsGAS.GASModel(dist, [true, true], 0.0, ["random walk slope", "random walk"], 
                                                            ["deterministic 12", "deterministic 12"], [missing, missing])
    fitted_model          = UnobservedComponentsGAS.fit(gas_model, y)
    output_initial_values = UnobservedComponentsGAS.create_output_initialization_from_fit(fitted_model, gas_model)

    @test(all(output_initial_values["rw"]["values"][:,1] .== 0))
    @test(all(output_initial_values["rw"]["values"][:,2] .!= 0))
    @test(all(output_initial_values["seasonality"]["values"] .!= 0))
    @test(all(output_initial_values["ar"]["values"] .== 0))
    @test(all(output_initial_values["slope"]["values"] .!= 0))
    @test(all(output_initial_values["rws"]["values"] .!= 0))

end