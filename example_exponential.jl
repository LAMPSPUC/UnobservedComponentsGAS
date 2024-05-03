import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/UnobservedComponentsGAS.jl")

using Random, Distributions, Plots, CSV, DataFrames, JuMP, Ipopt

path = "C:/Users/mathe/OneDrive/Documents/Academia/PUC/Mestrado/Dissertação/Dados/"
dados = CSV.read(path*"hsales.csv", DataFrame)
y = dados[:, 2]
T = length(y)

dist = UnobservedComponentsGAS.ExponentialDistribution();
d                   = 1.0;

function log_pdf_exp(θ, y)
     return Distributions.logpdf(Exponential(θ), y)
end

model = JuMP.Model(Ipopt.Optimizer)

#criando parametros da distribuição
@variable(model, θ[1:T] ≥ 1e-4)
parameters = Matrix(undef, T, 1)
parameters[:, 1] = model[:θ]

#criando variaveis da dinamica
s = UnobservedComponentsGAS.compute_score(model, parameters, y, d, time_varying_params, T, dist)
@variable(model, RW[1:T])
@variable(model, κ_RW)
@constraint(model, [t = 2:T], RW[t] == RW[t-1] + κ_RW * s[1][t])
@constraint(model,  1e-4 ≤ κ_RW)

@variable(model, γ[1:6])
@variable(model, γ_star[1:6])
@expression(model, S[t = 2:T], sum(γ[i]*cos(2 * π * i * t/12) + 
                         γ_star[i] * sin(2 * π * i* t/12) for i in 1:6))

#atribuindo a dinamica ao parametro de media
@expression(model, μ[t = 2:T], model[:RW][t] + model[:S][t])

#incluindo restrição que liga a média com o padrametro do distributions
# usar o DICT_MEAN_CONVERSION
@operator(model, get_mean, 1, UnobservedComponentsGAS.DICT_MEAN_CONVERSION["Exponential"])
@constraint(model, [t = 2:T], μ[t] == θ[t])

#incluindo funcao objetivo usando a log_pdf do distributions
@operator(model, log_pdf, 2, log_pdf_exp)
@objective(model, Max, sum(log_pdf(parameters[t, 1], y[t]) for t in 2:T))

optimize!(model)

fit = value.(μ).data

fitted_θ = value.(θ)

fitted_median   = Distributions.median.(Distributions.Exponential.(fitted_θ))
fitted_mode     = Distributions.mode.(Distributions.Exponential.(fitted_θ))
fitted_var      = Distributions.var.(Distributions.Exponential.(fitted_θ))
fitted_kurtosis = Distributions.kurtosis.(Distributions.Exponential.(fitted_θ))
fitted_skewness = Distributions.skewness.(Distributions.Exponential.(fitted_θ))
fitted_entropy  = Distributions.entropy.(Distributions.Exponential.(fitted_θ))

plot(y, label = "observed")
plot!(fit, label = "mean")
plot!(fitted_median, label = "median")
