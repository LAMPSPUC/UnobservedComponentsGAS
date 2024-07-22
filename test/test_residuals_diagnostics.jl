# Start defining the tests
@testset "Residuals Diagnostics Tests" begin

    time_series = CSV.read(joinpath(@__DIR__, "data/timeseries_lognormal_rws_d1.csv"), DataFrame)
    y = time_series[:,2]
    T = length(y)
    
    dist = UnobservedComponentsGAS.NormalDistribution()
    d    = 0.0

    params      = [true, false]
    level       = ["random walk slope", ""]
    seasonality = ["deterministic 12", ""]
    ar          = [1, missing]
    gas_model = UnobservedComponentsGAS.GASModel(dist, params, d, level, seasonality, ar)
    fitted_model = UnobservedComponentsGAS.fit(gas_model, y);

    @testset "get_residuals" begin
        @test length(UnobservedComponentsGAS.get_residuals(fitted_model; type="q")) == 99
        @test length(UnobservedComponentsGAS.get_residuals(fitted_model; type="std")) == 99
        @test size(UnobservedComponentsGAS.get_residuals(fitted_model; type="cs")) == (99, 2)
    end

    @testset "plot_residuals" begin
        @test UnobservedComponentsGAS.plot_residuals(fitted_model; type="q") isa Plots.Plots.Plot
        @test UnobservedComponentsGAS.plot_residuals(fitted_model; type="std") isa Plots.Plot
        @test UnobservedComponentsGAS.plot_residuals(fitted_model; type="cs") isa Plots.Plot
    end

    @testset "get_acf" begin
        acf_q = UnobservedComponentsGAS.get_acf_residuals(fitted_model; lags=25, type="q")
        acf_std = UnobservedComponentsGAS.get_acf_residuals(fitted_model; lags=25, type="std")
        @test length(acf_q) == 26
        @test length(acf_std) == 26
    end

    @testset "plot_acf_residuals" begin
        @test UnobservedComponentsGAS.plot_acf_residuals(fitted_model; lags=25, type="q") isa Plots.Plot
        @test UnobservedComponentsGAS.plot_acf_residuals(fitted_model; lags=25, type="std") isa Plots.Plot
    end

    @testset "plot_histogram" begin
        @test UnobservedComponentsGAS.plot_histogram(fitted_model; type="q", bins=20) isa Plots.Plot
        @test UnobservedComponentsGAS.plot_histogram(fitted_model; type="std", bins=20) isa Plots.Plot
    end

    @testset "plot_qqplot" begin
        @test UnobservedComponentsGAS.plot_qqplot(fitted_model; type="q") isa Plots.Plot
        @test UnobservedComponentsGAS.plot_qqplot(fitted_model; type="std") isa Plots.Plot
    end

    @testset "jarquebera" begin
        jb_q = UnobservedComponentsGAS.jarquebera(fitted_model; type="q")
        jb_std = UnobservedComponentsGAS.jarquebera(fitted_model; type="std")
        @test haskey(jb_q, "stat") && haskey(jb_q, "pvalue") && haskey(jb_q, "skew") && haskey(jb_q, "kurt")
        @test haskey(jb_std, "stat") && haskey(jb_std, "pvalue") && haskey(jb_std, "skew") && haskey(jb_std, "kurt")
    end

    @testset "ljungbox" begin
        lb_q = UnobservedComponentsGAS.ljungbox(fitted_model; type="q", lags=25)
        lb_std = UnobservedComponentsGAS.ljungbox(fitted_model; type="std", lags=25)
        @test haskey(lb_q, "stat") && haskey(lb_q, "pvalue")
        @test haskey(lb_std, "stat") && haskey(lb_std, "pvalue")
    end

    @testset "Htest" begin
        H_q = UnobservedComponentsGAS.Htest(fitted_model; type="q")
        H_std = UnobservedComponentsGAS.Htest(fitted_model; type="std")
        @test haskey(H_q, "stat") && haskey(H_q, "pvalue")
        @test haskey(H_std, "stat") && haskey(H_std, "pvalue")
    end

    @testset "archtest" begin
        arch_q = UnobservedComponentsGAS.archtest(fitted_model; type="q", lags=25)
        arch_std = UnobservedComponentsGAS.archtest(fitted_model; type="std", lags=25)
        @test haskey(arch_q, "stat") && haskey(arch_q, "pvalue")
        @test haskey(arch_std, "stat") && haskey(arch_std, "pvalue")
    end

    @testset "get_residuals_diagnosis_pvalues" begin
        diagnosis_pvalues_q = UnobservedComponentsGAS.get_residuals_diagnosis_pvalues(fitted_model; lags=25, type="q")
        diagnosis_pvalues_std = UnobservedComponentsGAS.get_residuals_diagnosis_pvalues(fitted_model; lags=25, type="std")
        @test haskey(diagnosis_pvalues_q, "JarqueBera") &&
              haskey(diagnosis_pvalues_q, "HVariance") &&
              haskey(diagnosis_pvalues_q, "LjungBox") &&
              haskey(diagnosis_pvalues_q, "LjungBoxSquared") &&
              haskey(diagnosis_pvalues_q, "ARCH")
        @test haskey(diagnosis_pvalues_std, "JarqueBera") &&
              haskey(diagnosis_pvalues_std, "HVariance") &&
              haskey(diagnosis_pvalues_std, "LjungBox") &&
              haskey(diagnosis_pvalues_std, "LjungBoxSquared") &&
              haskey(diagnosis_pvalues_std, "ARCH")
    end

end
