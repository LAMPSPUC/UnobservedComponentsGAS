"""
mutable struct NormalDistribution

    A mutable struct for representing the Normal distribution.

    # Fields
    - `μ::Union{Missing, Float64}`: Mean parameter.
    - `σ²::Union{Missing, Float64}`: Variance parameter.
"""
mutable struct NormalDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
end

"""
NormalDistribution()

Outer constructor for the Normal distribution, with no arguments specified.
    
    # Returns
    - The NormalDistribution struct with both fields set to missing.
"""
function NormalDistribution()
    return NormalDistribution(missing, missing)
end


"""
score_normal(μ, σ², y)

Compute the score vector of the Normal distribution, taking into account the specified parameters and observation.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    - `y`: The value of the observation.
    
    # Returns
    - A vector of type Float64, where the first element corresponds to the score related to the mean parameter, and the second element corresponds to the score related to the variance parameter.
"""
function score_normal(μ, σ², y) 
  
    return [(y - μ)/σ²; -(0.5/σ²) * (1 - ((y - μ)^2)/σ²)]
end


"""
fisher_information_normal(μ, σ²)

Compute the Fisher Information matrix of the Normal distribution, taking into account the specified parameters.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    
    # Returns
    - The Fisher Information matrix of the Normal distribution.
"""
function fisher_information_normal(μ, σ²)

    return [1/(σ²) 0; 0 1/(2*(σ²^2))]
end


"""
logpdf_normal(μ, σ², y)

Compute the logarithm of the probability density function (PDF) of the Normal distribution, considering the specified parameters and observation.
    # Arguments

    - `μ`: The value of the mean parameter.
    - `σ²`: The value of the variance parameter.
    - `y`: The value of the observation.
    
    # Returns
    - The value of the  logarithm of the probability density function (PDF) of the Normal distribution.
"""
function logpdf_normal(μ, σ², y)

    return logpdf_normal([μ, σ²], y)
end

"""
logpdf_normal(param, y)

Compute the logarithm of the probability density function (PDF) of the Normal distribution, considering the specified parameters and observation.
    # Arguments

    - `param`: The vector representing the parameters of the Normal distribution.
    - `y`: The value of the observation.
    
    # Returns
    - The value of the  logarithm of the probability density function (PDF) of the Normal distribution.
"""
function logpdf_normal(param, y)

    if param[2] < 0
        param[2] = 1e-4
    end

    return -0.5 * log(2 * π * param[2]) - ((y - param[1])^2)/(2 * param[2])
end


"""
cdf_normal(param::Vector{Float64}, y::Fl) where Fl

Compute the cumulative density function (CDF) of the Normal distribution, considering the specified parameters and observation.
    # Arguments

    - `param::Vector{Float64}`: The vector representing the parameters of the Normal distribution.
    - `y::Fl`: The value of the observation.
    
    # Returns
    - The value of the  cumulative density function (CDF) of the Normal distribution.
"""
function cdf_normal(param::Vector{Float64}, y::Fl) where Fl

    return Distributions.cdf(Normal(param[1], sqrt(param[2])), y)
end


"""
get_dist_code(dist::NormalDistribution)

 Provide the code representing the Normal distribution.
    
    # Arguments

    - `dist::NormalDistribution`: The structure that represents the Normal distribution.
    
    # Returns
    - The distribution code corresponding to the Normal distribution. (In this case, it always returns 1.)

"""
function get_dist_code(dist::NormalDistribution)
    return 1
end

"""
get_num_params(dist::NormalDistribution)

Provide the number of parameters for a Normal distribution.
    
    # Arguments

    - `dist::LogNormalDistribution`: The structure that represents the Normal distribution.
    
    # Returns
    - Int64(2)
"""
function get_num_params(dist::NormalDistribution)
    return 2
end

"""
sample_dist(param::Vector{Float64}, dist::NormalDistribution)

 Sample a random realization from a Normal distribution with the specified parameters.
    
    # Arguments

    -``param::Vector{Float64}`: A vector containing the parameters of the Normal distribution.
        - param[1] = μ
        - param[2] = σ²
    - `dist::NormalDistribution`: The structure that represents the Normal distribution.
    
    # Returns
    - A realization of N(param[1], √param[2])
"""
function sample_dist(param::Vector{Float64}, dist::NormalDistribution)

    if param[2] < 0.0
        param[2] = 1e-4
    end
    
    return rand(Normal(param[1], sqrt(param[2])))
end


"""
check_positive_constrainst(dist::NormalDistribution)

 Indicates which parameter of the Normal distribution must be positive.
    
    # Arguments

    - `dist::NormalDistribution`: The structure that represents the Normal distribution.
    
    # Returns
    - A boolean vector indicating that the variance parameter must be positive, while the mean parameter does not necessarily need to be positive.
"""
function check_positive_constrainst(dist::NormalDistribution)
    return [false, true]
end


"""
get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::NormalDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl
 
Compute the initial values for the parameters of the Normal distribution that will be used in the model's initialization.
    
    # Arguments

    - `y::Vector{Fl}`: The vector containing observed values.
    - `time_varying_params::Vector{Bool}`: The vector indicating which parameters will be treated as time-varying or fixed.
    - `dist::NormalDistribution`: The structure that represents the Normal distribution.
    - `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: The dictionary indicating whether each parameter has a seasonal component in its dynamics or not.
    
    # Returns
    - initial_params: Dict{Int64, Any}: Dictionary containing the initial values for each parameter.
        - 1: Initial values for the mean parameter, which can be fixed or time-varying.
        - 2: Initial values for the variance parameter, which can be fixed or time-varying.
"""
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::NormalDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl

    #T         = length(y)
    #dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict{Int64, Any}()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist) #(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = var(diff(y))
    end

    return initial_params
end

"""
get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::NormalDistribution) where Fl

Compute the variance of the observations for each seasonal period. For example, for monthly time series, calculate the variance of the observations for each month.

    # Arguments

    - `y::Vector{Fl}`: The vector containing observed values.
    - `seasonal_period::Int64`: The number that indicates the seasonal period of the varaince. 
    - `dist::NormalDistribution`: The structure that represents the Normal distribution.
    
    # Returns
    - seasonal_variances: Vector{Fl}: The vector containing the values of the seasonal variance.
"""
function get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::NormalDistribution) where Fl

    num_periods = ceil(Int, length(y) / seasonal_period)
    seasonal_variances = zeros(Fl, length(y))

    for i in 1:seasonal_period
        month_data = y[i:seasonal_period:end]
        num_observations = length(month_data)
        if num_observations > 0
            variance = Distributions.fit(Normal, month_data).σ^2
            
            for j in 1:num_observations
                seasonal_variances[i + (j - 1) * seasonal_period] = variance
            end
        end
    end
    return seasonal_variances
end
 
