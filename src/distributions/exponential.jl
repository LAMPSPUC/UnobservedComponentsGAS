"""
mutable struct ExponentialDistribution

    A mutable struct for representing the Exponential distribution.

    # Fields
    - `θ::Union{Missing, Float64}`: Mean parameter.
"""
mutable struct ExponentialDistribution <: ScoreDrivenDistribution
    θ::Union{Missing, Float64}
end

"""
ExponentialDistribution()

Outer constructor for the Exponential distribution, with no arguments specified.
    
    # Returns
    - The ExponentialDistribution struct with both fields set to missing.
"""
function ExponentialDistribution()
    return ExponentialDistribution(missing)
end


"""
score_exponential(θ , y)

Compute the score vector of the Exponential distribution, taking into account the specified parameters and observation.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    - `y`: The value of the observation.
    
    # Returns
    - A vector of type Float64, where the first element corresponds to the score related to the mean parameter, and the second element corresponds to the score related to the variance parameter.
"""
function score_exponential(θ_val, y_val) 
  
    # Symbolics.@variables y, θ

    # log_pdf_exp = logpdf_exponential(θ, y)

    # d_θ  = Symbolics.derivative(log_pdf_exp, θ)

    # d_θ_val = parse(Float64, substitute(d_θ, Dict(θ => θ_val, y => y_val)))
    # f_fun = eval(build_function(f, a,b))
    # return [d_θ_val]
    return [-1/θ_val + y_val/(θ_val^2)]
end


"""
fisher_information_exponential(μ, σ²)

Compute the Fisher Information matrix of the Exponential distribution, taking into account the specified parameters.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    
    # Returns
    - The Fisher Information matrix of the Exponential distribution.
"""
function fisher_information_exponential(θ)

    # Symbolics.@variables y, θ
    # pdf_exp = (1 / θ) * exp(-y/θ)
    # log_pdf_exp = - log(θ) -y/θ 

    # d_θ  = Symbolics.derivative(log_pdf_exp, θ)
    # d2_θ = Symbolics.derivative(d_θ , θ)

    # f = SymbolicNumericIntegration.integrate(-d2_θ * pdf_exp, (y, 0.0, 1e6); detailed = false)
    # f_val = Float64(substitute(f, Dict(θ => θ_val)))
    # return [f_val]
    return [1/(θ^2)]
end

" Returns the distribution mean, given the specified parameter"
function mean_conversion_exponential(θ)
    return θ
    # return mean(dist) # opção 2, se passar a Exponencial(θ) para a mean, ela retorna o valor da média, θ.
    # Talvez usar isso para generalizar as expressões de média de outras distribuições.
    # Ex: dist = Gamma(α=10,θ=2) -> mean(dist) retorna α*θ = 20
end


"""
logpdf_exponential(μ, σ², y)

Compute the logarithm of the probability density function (PDF) of the Exponential distribution, considering the specified parameters and observation.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    - `y`: The value of the observation.
    
    # Returns
    - The value of the  logarithm of the probability density function (PDF) of the Exponential distribution.
"""
function logpdf_exponential(θ, y)

    return logpdf_exponential([θ], y)
end

"""
logpdf_exponential(param, y)

Compute the logarithm of the probability density function (PDF) of the Exponential distribution, considering the specified parameters and observation.
    # Arguments

    - `param`: The vector representing the parameters of the Exponential distribution.
    - `y`: The value of the observation.
    
    # Returns
    - The value of the  logarithm of the probability density function (PDF) of the Exponential distribution.
"""
function logpdf_exponential(param, y)

    if typeof(param[1]) != Num
        if param[1] < 0
            param[1] = 1e-4
        end
    end

    return -log(param[1]) -y/param[1] 
end

"""
cdf_exponential(param::Vector{Float64}, y::Fl) where Fl

Compute the cumulative density function (CDF) of the Exponential distribution, considering the specified parameters and observation.
    # Arguments

    - `param::Vector{Float64}`: The vector representing the parameters of the Exponential distribution.
    - `y::Fl`: The value of the observation.
    
    # Returns
    - The value of the  cumulative density function (CDF) of the Exponential distribution.
"""
function cdf_exponential(param::Vector{Float64}, y::Fl) where Fl

    return Distributions.cdf(Exponential(param[1]), y)
end


"""
get_dist_code(dist::ExponentialDistribution)

 Provide the code representing the Exponential distribution.
    
    # Arguments

    - `dist::ExponentialDistribution`: The structure that represents the Exponential distribution.
    
    # Returns
    - The distribution code corresponding to the Exponential distribution. (In this case, it always returns 1.)

"""
function get_dist_code(dist::ExponentialDistribution)
    return 3
end

"""
get_num_params(dist::ExponentialDistribution)

Provide the number of parameters for a Exponential distribution.
    
    # Arguments

    - `dist::LogExponentialDistribution`: The structure that represents the Exponential distribution.
    
    # Returns
    - Int64(2)
"""
function get_num_params(dist::ExponentialDistribution)
    return 1
end

"""
sample_dist(param::Vector{Float64}, dist::ExponentialDistribution)

 Sample a random realization from a Exponential distribution with the specified parameters.
    
    # Arguments

    -``param::Vector{Float64}`: A vector containing the parameters of the Exponential distribution.
        - param[1] = μ
        - param[2] = σ²
    - `dist::ExponentialDistribution`: The structure that represents the Exponential distribution.
    
    # Returns
    - A realization of N(param[1], √param[2])
"""
function sample_dist(param::Vector{Float64}, dist::ExponentialDistribution)

    if param[1] < 0.0
        param[1] = 1e-4
    end
    
    return rand(Exponential(param[1]))
end


"""
check_positive_constrainst(dist::ExponentialDistribution)

 Indicates which parameter of the Exponential distribution must be positive.
    
    # Arguments

    - `dist::ExponentialDistribution`: The structure that represents the Exponential distribution.
    
    # Returns
    - A boolean vector indicating that the variance parameter must be positive, while the mean parameter does not necessarily need to be positive.
"""
function check_positive_constrainst(dist::ExponentialDistribution)
    return [true]
end


"""
get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::ExponentialDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl
 
Compute the initial values for the parameters of the Exponential distribution that will be used in the model's initialization.
    
    # Arguments

    - `y::Vector{Fl}`: The vector containing observed values.
    - `time_varying_params::Vector{Bool}`: The vector indicating which parameters will be treated as time-varying or fixed.
    - `dist::ExponentialDistribution`: The structure that represents the Exponential distribution.
    - `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: The dictionary indicating whether each parameter has a seasonal component in its dynamics or not.
    
    # Returns
    - initial_params: Dict{Int64, Any}: Dictionary containing the initial values for each parameter.
        - 1: Initial values for the mean parameter, which can be fixed or time-varying.
        - 2: Initial values for the variance parameter, which can be fixed or time-varying.
"""
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::ExponentialDistribution, seasonality::Dict{Int64, Union{Bool, Int64}}) where Fl

    #T         = length(y)
    #dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict{Int64, Any}()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    return initial_params
end
