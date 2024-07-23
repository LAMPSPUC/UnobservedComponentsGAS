@testset "Components Dynamics" begin
    
    @testset "has_dynamic" begin
        @test(UnobservedComponentsGAS.has_random_walk(["random walk"]))
        @test(!UnobservedComponentsGAS.has_random_walk(["random walk slope"]))
        @test(UnobservedComponentsGAS.has_random_walk_slope(["random walk slope"]))
        @test(!UnobservedComponentsGAS.has_random_walk_slope(["random walk"]))
        @test(UnobservedComponentsGAS.has_ar1_level(["ar(1)",""]))
        @test(!UnobservedComponentsGAS.has_ar1_level([""]))
        @test(UnobservedComponentsGAS.has_seasonality(["deterministic 12"]))
        @test(!UnobservedComponentsGAS.has_seasonality(""))
        @test(UnobservedComponentsGAS.has_AR(1))
        @test(!UnobservedComponentsGAS.has_AR(missing))


        @test(UnobservedComponentsGAS.has_random_walk(["random walk", ""], 1))
        @test(UnobservedComponentsGAS.has_random_walk(["", "random walk"], 2))
        @test(UnobservedComponentsGAS.has_random_walk_slope(["random walk slope",""], 1))
        @test(UnobservedComponentsGAS.has_random_walk_slope(["","random walk slope"], 2))
        @test(UnobservedComponentsGAS.has_ar1_level(["ar(1)",""], 1))
        @test(UnobservedComponentsGAS.has_ar1_level(["","ar(1)"], 2))
        @test(UnobservedComponentsGAS.has_seasonality(["deterministic 12", ""], 1))
        @test(!UnobservedComponentsGAS.has_seasonality(["deterministic 12",""], 2))
        @test(UnobservedComponentsGAS.has_AR(1, 1))
        @test(!UnobservedComponentsGAS.has_AR([1,missing], 2))

        @test(isequal(UnobservedComponentsGAS.get_AR_order([2,0]), [[1, 2], [nothing]]))
        @test(isequal(UnobservedComponentsGAS.get_AR_order([missing, missing]), [[nothing], [nothing]]))
    end

    @testset "Number of variables and constraints" begin
        T = 100
        s = [zeros(T), zeros(T)]

        model = JuMP.Model(Ipopt.Optimizer)
        @test(num_variables(model) == 0)
            
        # Test number of variables and constraints of AR
        order                    = [2, 1]
        number_of_variables_ar   = (T + maximum(order) + 1) * sum(.!iszero.(order))
        number_of_constraints_ar = (T - maximum(order) + 1 ) * sum(.!iszero.(order)) + maximum(order) - minimum(order)
        UnobservedComponentsGAS.add_AR!(model, s, T, order) 
        @test(num_variables(model) == number_of_variables_ar)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_ar)

        # Test number of variables and constraints of RW
        model                    = JuMP.Model(Ipopt.Optimizer)
        rw                       = [true, false]
        number_of_variables_rw   = (T + 1)*rw[1] + (T + 1)*rw[2]
        number_of_constraints_rw = (T - 1 + rw[1])*rw[1] + (T - 1 + rw[2])*rw[2]
        UnobservedComponentsGAS.add_random_walk!(model, s, T, Dict(1=>rw[1], 2=>rw[2]))  #ta dando um bounds error
        @test(num_variables(model) == number_of_variables_rw)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_rw)

        # Test number of variables and constraints of RWS
        model                     = JuMP.Model(Ipopt.Optimizer)
        rws                       = [true, false]
        number_of_variables_rws   = (2*T + 2) * sum(rws)
        number_of_constraints_rws = (2*(T-1) + 2) * sum(rws)
        UnobservedComponentsGAS.add_random_walk_slope!(model, s, T, Dict(1=>rws[1], 2=>rws[2]))  #ta dando um bounds error
        @test(num_variables(model) == number_of_variables_rws)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_rws)

        # Test number of variables and constraints of Seasonality
        seasonal_periods = 12
        seasonality      = ["deterministic 12"]
        seasonality_dict, stochastic = UnobservedComponentsGAS.get_seasonality_dict_and_stochastic(seasonality)

        num_harmonic, seasonal_period = UnobservedComponentsGAS.get_num_harmonic_and_seasonal_period(seasonality_dict)
        @test((num_harmonic, seasonal_period) == ([Int64(floor(seasonal_periods))/2], [seasonal_periods]))
        
        # Deterministic
        model                    = JuMP.Model(Ipopt.Optimizer)
        idx_params               = findall(i -> i != false, seasonality)
        unique_num_harmonic      = unique(num_harmonic)[minimum(idx_params)]
        number_of_variables_sd   = 2*unique_num_harmonic
        number_of_constraints_sd = 0
        UnobservedComponentsGAS.add_trigonometric_seasonality!(model, s, T, seasonality)
        @test(num_variables(model) == number_of_variables_sd)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_sd)

        # and stochastic
        model                    = JuMP.Model(Ipopt.Optimizer)
        number_of_variables_ss   = 1 + 12*T
        number_of_constraints_ss = 1 + (T-1)*unique_num_harmonic*2
        seasonality      = ["stochastic 12"]
        UnobservedComponentsGAS.add_trigonometric_seasonality!(model, s, T, seasonality)
        @test(num_variables(model) == number_of_variables_ss)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_ss)
        
        # Test number of variables and constraints from include_components! with Deterministic Seasonality
        dist      = UnobservedComponentsGAS.NormalDistribution()
        gas_model = UnobservedComponentsGAS.GASModel(dist, [true, true], 1.0, ["random walk slope", ""], ["deterministic 12", ""], order)

        model = JuMP.Model(Ipopt.Optimizer)         
        UnobservedComponentsGAS.include_components!(model, s, gas_model, T)
        @test(num_variables(model) == number_of_variables_ar + 
                                    number_of_variables_rws + 
                                    number_of_variables_sd)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_ar + 
                                                                                number_of_constraints_rws + 
                                                                                number_of_constraints_sd)
        
        # Test number of variables and constraints from include_components! with stochastic Seasonality
        gas_model = UnobservedComponentsGAS.GASModel(dist, [true, true], 1.0, ["random walk slope", ""], ["stochastic 12", ""], [2, 1])

        model = JuMP.Model(Ipopt.Optimizer)         
        UnobservedComponentsGAS.include_components!(model, s, gas_model, T)
        @test(num_variables(model) == number_of_variables_ar + 
                                    number_of_variables_rws + 
                                    number_of_variables_ss)
        @test(num_constraints(model; count_variable_in_set_constraints=true) == number_of_constraints_ar + 
                                                                                number_of_constraints_rws + 
                                                                                number_of_constraints_ss)
    end

end