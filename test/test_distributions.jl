@testset "Distributions" begin
    
    "FALTAM TESTES RELATIVOS ÀS EXPLICATIVAS"

    T = 100
    Random.seed!(1234)
    y = rand(T)
    X = [2*y y/2 rand(T)]
    

    @info(" --- Test distributions/common")
    μ  = 1.
    σ² = 2.
    ν  = 10

    "Test NormalDistribution scaled_score"
    scaled_score_normal_0  = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 0.0, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.NormalDistribution()), 1)
    scaled_score_normal_05 = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 0.5, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.NormalDistribution()), 1)
    scaled_score_normal_1  = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 1.0, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.NormalDistribution()), 1)

    @test(round(scaled_score_normal_0, digits = 3)  == -0.21)
    @test(round(scaled_score_normal_05, digits = 3) == -0.297)
    @test(round(scaled_score_normal_1, digits = 3)  == -0.42)

    "Test LogNormalDistribution scaled_score"
    scaled_score_lognormal_0  = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 0.0, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.LogNormalDistribution()), 1)
    scaled_score_lognormal_05 = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 0.5, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.LogNormalDistribution()), 1)
    scaled_score_lognormal_1  = UnobservedComponentsGAS.scaled_score(μ, σ², y[1], 1.0, 
                                                                UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.LogNormalDistribution()), 1)

    @test(round(scaled_score_lognormal_0, digits = 3)  == -0.21)
    @test(round(scaled_score_lognormal_05, digits = 3) == -0.297)
    @test(round(scaled_score_lognormal_1, digits = 3)  == -0.42)

    "Test tLocationScaleDistribution scaled_score"
    scaled_score_t_0  = UnobservedComponentsGAS.scaled_score(μ, σ², ν, y[1], 0.0, 
                                                            UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.tLocationScaleDistribution()), 1)
    scaled_score_t_05 = UnobservedComponentsGAS.scaled_score(μ, σ², ν, y[1], 0.5, 
                                                            UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.tLocationScaleDistribution()), 1)
    scaled_score_t_1  = UnobservedComponentsGAS.scaled_score(μ, σ², ν, y[1], 1.0, 
                                                            UnobservedComponentsGAS.get_dist_code(UnobservedComponentsGAS.tLocationScaleDistribution()), 1)

    @test(round(scaled_score_t_0, digits = 3)  == -0.229)
    @test(round(scaled_score_t_05, digits = 3) == -0.352)
    @test(round(scaled_score_t_1, digits = 3)  == -0.541)

    @info(" --- Test distributions/normal")
    dist            = UnobservedComponentsGAS.NormalDistribution()
    seasonal_period = 12
    μ               = 0.
    σ²              = 1.
    score_normal_1          = UnobservedComponentsGAS.score_normal(μ, σ² ,0)
    score_normal_2          = UnobservedComponentsGAS.score_normal(μ, σ² ,1)
    fisher_normal           = UnobservedComponentsGAS.fisher_information_normal(μ, σ²)
    logpdf_normal_1         = UnobservedComponentsGAS.logpdf_normal(μ, σ², 0)
    logpdf_normal_2         = UnobservedComponentsGAS.logpdf_normal(μ, σ², 1)
    cdf_normal_1            = UnobservedComponentsGAS.cdf_normal([μ, σ²], 0)
    cdf_normal_2            = UnobservedComponentsGAS.cdf_normal([μ, σ²], 1)
    initial_params_normal_1 = UnobservedComponentsGAS.get_initial_params(y, [true, false], dist, Dict{Int64, Union{Bool, Int64}}(1=>12))
    initial_params_normal_2 = UnobservedComponentsGAS.get_initial_params(y, [false, false], dist, Dict{Int64, Union{Bool, Int64}}(1=>12))
    seasonal_variances      = UnobservedComponentsGAS.get_seasonal_var(y,seasonal_period, dist)


    @test(size(score_normal_1) == (2,))
    @test(all(score_normal_1 .== [0., -0.5]))
    @test(all(score_normal_2 .== [1., -0.0]))
    @test(all(fisher_normal .== [1. 0.; 0. 0.5]))
    @test(round(logpdf_normal_1, digits = 3) == -0.919)
    @test(round(logpdf_normal_2, digits = 3) == -1.419)
    @test(round(cdf_normal_1, digits = 3) == 0.5)
    @test(round(cdf_normal_2, digits = 3) == 0.841)
    @test(UnobservedComponentsGAS.get_dist_code(dist) == 1)
    @test(UnobservedComponentsGAS.get_num_params(dist) == 2)
    @test(UnobservedComponentsGAS.check_positive_constrainst(dist) == [false, true])
    @test(initial_params_normal_1[1] == y)
    @test(initial_params_normal_1[2] == var(diff(y)))
    @test(initial_params_normal_2[1] == mean(y))
    @test(all(seasonal_variances .> 0))

    @info(" --- Test distributions/t_location_scale")
    dist            = UnobservedComponentsGAS.tLocationScaleDistribution()
    seasonal_period = 12
    μ               = 0.
    σ²              = 1.
    ν               = 1
    score_tlocationscale_1          = UnobservedComponentsGAS.score_tlocationscale(μ, σ², ν ,0)
    score_tlocationscale_2          = UnobservedComponentsGAS.score_tlocationscale(μ, σ², ν ,1)
    fisher_tlocationscale           = UnobservedComponentsGAS.fisher_information_tlocationscale(μ, σ², ν)
    logpdf_tlocationscale_1         = UnobservedComponentsGAS.logpdf_tlocationscale(μ, σ², ν, 0)
    logpdf_tlocationscale_2         = UnobservedComponentsGAS.logpdf_tlocationscale(μ, σ², ν, 1)
    cdf_tlocationscale_1            = UnobservedComponentsGAS.cdf_tlocationscale([μ, σ², ν], 0)
    cdf_tlocationscale_2            = UnobservedComponentsGAS.cdf_tlocationscale([μ, σ², ν], 1)
    initial_params_tlocationscale_1 = UnobservedComponentsGAS.get_initial_params(y, [true, false, false], dist, Dict{Int64, Union{Bool, Int64}}(1=>12))
    initial_params_tlocationscale_2 = UnobservedComponentsGAS.get_initial_params(y, [false, false, true], dist, Dict{Int64, Union{Bool, Int64}}(1=>12))
    seasonal_variances              = UnobservedComponentsGAS.get_seasonal_var(y,seasonal_period, dist)

    gas_model = UnobservedComponentsGAS.GASModel(dist, [true, false, false], 0.0, ["random walk", "", ""], "deterministic 12", missing)    
    
    gas_model_2 = deepcopy(gas_model)
    best_model_no_explanatory, best_ν_no_explanatory = UnobservedComponentsGAS.find_first_model_for_local_search(gas_model, y)

    # Problema quando tiramos o pacote do André para selecionar as explicativas
    # best_model_explanatory, best_ν_explanatory       = UnobservedComponentsGAS.find_first_model_for_local_search(gas_model, y, X)
    
    @test(size(score_tlocationscale_1) == (2,))
    @test(all(score_tlocationscale_1 .== [0., -0.5]))
    @test(all(score_tlocationscale_2 .== [1., -0.0]))
    @test(all(fisher_tlocationscale .== [0.5 0.; 0. 0.125]))
    @test(round(logpdf_tlocationscale_1, digits = 3) == -1.145)
    @test(round(logpdf_tlocationscale_2, digits = 3) == -1.838)
    @test(round(cdf_tlocationscale_1, digits = 3) == 0.5)
    @test(round(cdf_tlocationscale_2, digits = 3) == 0.75)
    @test(UnobservedComponentsGAS.get_dist_code(dist) == 2)
    @test(UnobservedComponentsGAS.get_num_params(dist) == 3)
    @test(UnobservedComponentsGAS.check_positive_constrainst(dist) == [false, true, true])
    @test(initial_params_tlocationscale_1[1] == y)
    @test(initial_params_tlocationscale_1[2] == var(diff(y)))
    @test(initial_params_tlocationscale_1[3] == T-1)
    @test(initial_params_tlocationscale_2[1] == mean(y))
    @test(initial_params_tlocationscale_2[3] == (y.^2) ./ (ones(T) * var(diff(y))))
    @test(all(seasonal_variances .> 0))
    @test(all(best_ν_no_explanatory .== best_model_no_explanatory.fitted_params["param_3"]))
    

    @info(" --- Test distributions/log_normal")
    dist              = UnobservedComponentsGAS.LogNormalDistribution()
    param_1_exp       = zeros(T)
    param_1_log       = ones(T)
    param_2           = ones(T)
    fitted_params_exp = Dict("param_1" => param_1_exp,
                        "param_2" => param_2)
    fitted_params_log = Dict("param_1" => param_1_log,
                        "param_2" => param_2)
                      
                        
    new_fit_in_sample_exp, new_fitted_params_exp = UnobservedComponentsGAS.convert_to_exp_scale(y, fitted_params_exp)
    new_fitted_params_log = UnobservedComponentsGAS.convert_to_log_scale(fitted_params_log)

    @test(all(new_fit_in_sample_exp .> 0 ))
    @test(all(new_fit_in_sample_exp .== exp.(y .+ param_2./2)))
    @test(all(round.(new_fitted_params_exp["param_1"], digits = 3) .== round.(exp.(param_1_exp + param_2./2), digits=3)))
    @test(all(round.(new_fitted_params_exp["param_2"], digits = 3) .== round.(exp.(2*param_1_exp .+ param_2 ) .* (exp.(param_2) .- 1), digits=3)))
    @test(all(round.(new_fitted_params_log["param_1"], digits = 3) .== round.(log.(param_1_log) .- 0.5 .* log.(1 .+ param_2./(param_1_log.^2)), digits=3)))
    @test(all(round.(new_fitted_params_log["param_2"], digits = 3) .== round.(log.(1 .+ param_2./(param_1_log.^2)), digits=3)))    
    @test(UnobservedComponentsGAS.get_dist_code(dist) == 1)
    @test(UnobservedComponentsGAS.get_num_params(dist) == 2)

end