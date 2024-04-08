@testset "Optimization" begin
    
    T = 100

    # Test include_parameters for Normal Distribution case 1
    time_varying_params = [true, true]
    dist                = UnobservedComponentsGAS.NormalDistribution()
    model               = JuMP.Model(Ipopt.Optimizer)
    params              = UnobservedComponentsGAS.include_parameters(model, time_varying_params, T, dist, missing)
    @test(size(params) == (T, 2))
    @test(sum((occursin.("fixed", JuMP.name.(params[:,1])))) == 0)
    @test(sum((occursin.("fixed", JuMP.name.(params[:,2])))) == 0)

    # Test include_parameters for Normal Distribution case 2
    time_varying_params = [true, false]
    model               = JuMP.Model(Ipopt.Optimizer)
    params              = UnobservedComponentsGAS.include_parameters(model, time_varying_params, T, dist, missing)
    @test(size(params) == (T, 2))
    @test(sum((occursin.("fixed", JuMP.name.(params[:,1])))) == 0)
    @test(sum((occursin.("fixed", JuMP.name.(params[:,2])))) == T)

    # Test compute_score for Normal Distribution
    model          = JuMP.Model(Ipopt.Optimizer)
    computed_score = UnobservedComponentsGAS.compute_score(model, params, zeros(T), 0.0, time_varying_params, T, dist)
    @test(size(computed_score) == (2,))
    @test(length(computed_score[1]) == T - 1)

    # Test include_parameters for tLocationScale Distribution
    time_varying_params = [true, true, false]
    dist                = UnobservedComponentsGAS.tLocationScaleDistribution()
    model               = JuMP.Model(Ipopt.Optimizer)
    params              = UnobservedComponentsGAS.include_parameters(model, time_varying_params, T, dist, 1)
    @test(size(params) == (T, 3))
    @test(sum((occursin.("fixed", JuMP.name.(params[:,1])))) == 0)
    @test(sum((occursin.("fixed", JuMP.name.(params[:,2])))) == 0)
    @test(sum((occursin.("fixed", JuMP.name.(params[:,3])))) == T)

    # Test compute_score for tLocationScale Distribution
    model          = JuMP.Model(Ipopt.Optimizer)
    computed_score = UnobservedComponentsGAS.compute_score(model, params, zeros(T), 0.0, time_varying_params, T, dist)
    @test(size(computed_score) == (3,))
    @test(length(computed_score[1]) == T - 1)

    # Test include_explanatory_variables!
    model  = JuMP.Model(Ipopt.Optimizer)
    X      = zeros(T,3)
    UnobservedComponentsGAS.include_explanatory_variables!(model, X)
    @test(num_variables(model) == 3)

    # Test include_objective_function! Normal Distribution case 1
    dist                = UnobservedComponentsGAS.NormalDistribution()
    dist_code           = UnobservedComponentsGAS.get_dist_code(dist)
    dist_name           = UnobservedComponentsGAS.DICT_CODE[dist_code]
    model               = JuMP.Model(Ipopt.Optimizer)
    time_varying_params = [true, false]
    params              = UnobservedComponentsGAS.include_parameters(model, time_varying_params, T, dist, missing)
    UnobservedComponentsGAS.include_objective_function!(model, params, zeros(T), T, false, dist_code) 
    @test(objective_function(model) == 0)

    # Test include_objective_function! Normal Distribution case 2
    model               = JuMP.Model(Ipopt.Optimizer)
    UnobservedComponentsGAS.include_objective_function!(model, params, zeros(T), T, true, dist_code) 
    @test(objective_function(model) == 0)

    # Test include_objective_function! tLocationScale Distribution case 1
    dist                = UnobservedComponentsGAS.tLocationScaleDistribution()
    dist_code           = UnobservedComponentsGAS.get_dist_code(dist)
    dist_name           = UnobservedComponentsGAS.DICT_CODE[dist_code]
    model               = JuMP.Model(Ipopt.Optimizer)
    time_varying_params = [true, true, false]
    params              = UnobservedComponentsGAS.include_parameters(model, time_varying_params, T, dist, 1)
    UnobservedComponentsGAS.include_objective_function!(model, params, zeros(T), T, false, dist_code) 
    @test(objective_function(model) == 0)

    # Test include_objective_function! tLocationScale Distribution case 2
    model               = JuMP.Model(Ipopt.Optimizer)
    UnobservedComponentsGAS.include_objective_function!(model, params, zeros(T), T, false, dist_code)
    @test(objective_function(model) == 0) 
end