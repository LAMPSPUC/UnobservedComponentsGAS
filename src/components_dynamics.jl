"""
has_random_walk(random_walk::Dict{Int64, Bool})   

Checks if there is any random walk component in the provided dictionary.

## Arguments
    - `random_walk::Dict{Int64, Bool}`: Dictionary indicating the presence of random walk components, where the keys represent indices and the values represent the presence of a random walk at each index.

## Returns
    - `true` if there is at least one random walk component, `false` otherwise.
"""
has_random_walk(random_walk::Dict{Int64, Bool}) = any(values(random_walk))


"""
has_random_walk_slope(random_walk_slope::Dict{Int64, Bool}) 

Checks if there is any random walk + slope dynamic in the provided dictionary.

## Arguments
    - `random_walk_slope::Dict{Int64, Bool}`: Dictionary indicating the presence of random walk + slope dynamic, where the keys represent indices and the values represent the presence of a random walk at each index.

## Returns
    - `true` if there is at least one component with a random walk + slope dynamic, `false` otherwise.
"""
has_random_walk_slope(random_walk_slope::Dict{Int64, Bool}) = any(values(random_walk_slope))


"""
has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}})

Checks if there is any seasonal component in the provided dictionary.

## Arguments
- `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: A dictionary indicating the presence of seasonality. If seasonality is defined, the value of the dictionary represents the seasonal period considered.

## Returns
- `true` if there is at least one component with seasonality, `false` otherwise.
"""
has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) = !all(isequal.(typeof.(values(seasonality)), Bool))


"""
# has_AR(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}})

Checks if the provided autoregressive (AR) dictionary indicates the presence of autoregressive components.

## Arguments
- `ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}`: A dictionary indicating the presence of autoregressive components. If autoregressive components are defined, the value of the dictionary varies based on the specific AR structure.

## Returns
- `true` if there is at least one component with autoregressive structure, `false` otherwise.
"""
has_AR(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}) = !all(isequal.(typeof.(values(ar)), Bool))


"""
# has_random_walk(random_walk::Dict{Int64, Bool}, param::Int64)

Checks if the specified parameter has a random walk component in the given dictionary.

## Arguments
- `random_walk::Dict{Int64, Bool}`: A dictionary indicating the presence of random walk components for different parameters.
- `param::Int64`: The parameter index to check for the presence of a random walk component.

## Returns
- `true` if the specified parameter has a random walk component, `false` otherwise.
"""
has_random_walk(random_walk::Dict{Int64, Bool}, param::Int64) = random_walk[param]

"""
# has_random_walk_slope(random_walk_slope::Dict{Int64, Bool}, param::Int64)

Checks if the specified parameter has a random walk with slope component in the given dictionary.

## Arguments

- `random_walk_slope::Dict{Int64, Bool}`: A dictionary indicating the presence of random walk with slope components for different parameters.
- `param::Int64`: The parameter index to check for the presence of a random walk with slope component.

## Returns
- `true` if the specified parameter has a random walk with slope component, `false` otherwise.
"""
has_random_walk_slope(random_walk_slope::Dict{Int64, Bool}, param::Int64) = random_walk_slope[param]


"""
# has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, param::Int64)

Checks if the specified parameter has a seasonality component in the given dictionary.

## Arguments

- `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: A dictionary indicating the presence of seasonality for different parameters. If the seasonal component is defined, the value of the dictionary indicates the seasonal period considered.
- `param::Int64`: The parameter index to check for the presence of a seasonality component.

## Returns
- `true` if the specified parameter has a seasonality component, `false` otherwise.
"""
has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, param::Int64) = seasonality[param] != false 

"""
# has_AR

Checks if the specified parameter has an autoregressive (AR) component in the given dictionary.

## Arguments
- `ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}`: A dictionary indicating the presence of autoregressive (AR) components for different parameters. The value of the dictionary can be either an integer indicating the order of the AR process, a vector specifying the AR lags, a boolean indicating the presence of AR, or any other type.
- `param::Int64`: The parameter index to check for the presence of an autoregressive (AR) component.

## Returns
- `true` if the specified parameter has an autoregressive (AR) component, `false` otherwise.
"""
has_AR(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}, param::Int64) = typeof(ar[param]) == Bool || ar[param] == 0 ? false : true

"""
# get_AR_order(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}})

Extracts the autoregressive (AR) orders for each parameter from the given dictionary.

## Arguments
- `ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}`: 
       A dictionary indicating the presence of autoregressive (AR) components for different parameters. 
       The value of the dictionary can be either an integer indicating the order of the AR process, 
       a vector specifying the AR lags, or a boolean indicating the absence of the AR component.

## Returns
- An array of vectors, where each vector contains the autoregressive (AR) orders corresponding to each parameter. If no AR component is present for a parameter, the order vector will contain a single element `[nothing]`.
"""

# To do: Troquei o retorno da função caso não tenha AR para nothing. Lembrar de trocar isso nos if's de outras funções.
function get_AR_order(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}})
    
    num_params = length(ar)
    order      = Vector{Union{Int64, Nothing}}[]

    for i in 1:num_params
        if typeof(ar[i]) == Int64
            push!(order, Int64.(collect(1:ar[i])))
        elseif typeof(ar[i]) == Vector{Int64}
            push!(order, Int64.(ar[i]))
        else
            push!(order, [nothing])
        end
    end
    return order
end

"""
# add_AR!(model::Ml, s::Vector{Fl}, T::Int64, ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}) where {Ml, Fl}

Incorporate the autoregressive (AR) component into the dynamics of the specified parameters within a predefined model.

## Arguments
- `model::Ml`: The model to which autoregressive (AR) components will be added.
- `s::Vector{Fl}`: The vector containing the scaled score expressions.
- `T::Int64`: The length of the time series.
- `ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}`: A dictionary indicating the presence of autoregressive (AR) components for different parameters. The value of the dictionary can be either an integer indicating the order of the AR process, a vector specifying the AR lags, or a boolean indicating the absence of the AR component.

## Returns
- Modifies the input model by adding autoregressive (AR) dynamics.
"""
function add_AR!(model::Ml, s::Vector{Fl}, T::Int64, ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}) where {Ml, Fl}

    idx_params = findall(i -> i != 0, ar) # Time-varying parameters with autoregressive dynamic
    order      = get_AR_order(ar)

    max_order     = maximum(vcat(order...)) # Maximum lag in the model
    unique_orders = filter(x -> x !isnothing(x), unique(vcat(order...)))#[findall(i -> i != 0.0, vcat(order...))]
    
    @variable(model, AR[1:T, idx_params])
    @variable(model, ϕ[1:max_order, idx_params])
    @variable(model, κ_AR[idx_params])

    @constraint(model, [i in idx_params], 1e-4 ≤ κ_AR[i])

    # Revisar essas restrições com o Alves !!
    for i in unique_orders
        for j in idx_params
            if i ∉ order[j]
                #@constraint(model, ϕ[i, idx_params[j]] == 0)
                JuMP.fix(model[:ϕ][i, j], 0.0)
            end
        end
    end

    @NLconstraint(model, [t = (max_order + 1):T, j in idx_params], AR[t, j] == sum(ϕ[p, j] * AR[t - p, j] for p in unique_orders) + κ_AR[j] * s[j][t])
end

"""
# add_random_walk_slope!(model::Ml, s::Vector{Fl}, T::Int64, random_walk_slope::Dict{Int64, Bool}) where {Ml, Fl}

Incorporate the random walk with slope component into the dynamics of the specified parameters within a predefined model.

## Arguments
- `model::Ml`: The model to which random walk with slope components will be added.
- `s::Vector{Fl}`: The vector containing the scaled score expressions.
- `T::Int64`: The length of the time series.
- `random_walk_slope::Dict{Int64, Bool}`: A dictionary indicating the presence of random walk with slope components for different parameters. The value of the dictionary indicates whether each parameter has a random walk with slope dynamic (`true` if it does, `false` otherwise).

## Returns
- Modifies the input model by adding random walk with slope components.
"""
function add_random_walk_slope!(model::Ml, s::Vector{Fl}, T::Int64, random_walk_slope::Dict{Int64, Bool}) where {Ml, Fl}
    
    idx_params = findall(i -> i == true, random_walk_slope) # Time-varying parameters with the random walk with slope dynamic

    @variable(model, RWS[1:T, idx_params])
    @variable(model, b[1:T, idx_params])
    @variable(model, κ_RWS[idx_params])
    @variable(model, κ_b[idx_params])

    @NLconstraint(model, [t = 2:T, j in idx_params], b[t, j] == b[t - 1, j] + κ_b[j] * s[j][t])
    @NLconstraint(model, [t = 2:T, j in idx_params], RWS[t, j] == RWS[t - 1, j] + b[t - 1, j] + κ_RWS[j] * s[j][t])
    @constraint(model, [j in idx_params], 1e-4 ≤ κ_RWS[j])
    @constraint(model, [j in idx_params], 1e-4 ≤ κ_b[j])
end

"""
# add_random_walk!(model::Ml, s::Vector{Fl}, T::Int64, random_walk::Dict{Int64, Bool}) where {Ml, Fl}

Incorporate the random walk component into the dynamics of the specified parameters within a predefined model.

## Arguments
- `model::Ml`: The model to which random walk components will be added.
- `s::Vector{Fl}`: The vector containing the scaled score expressions.
- `T::Int64`: The length of the time series.
- `random_walk::Dict{Int64, Bool}`: A dictionary indicating the presence of random walk components for different parameters. The value of the dictionary indicates whether each parameter has a random walk dynamic (`true` if it does, `false` otherwise).

## Returns
- Modifies the input model by adding random walk components.
"""
function add_random_walk!(model::Ml, s::Vector{Fl}, T::Int64, random_walk::Dict{Int64, Bool}) where {Ml, Fl}

    idx_params = findall(i -> i == true, random_walk) # Time-varying parameters with the random walk dynamic

    @variable(model, RW[1:T, idx_params])
    @variable(model, κ_RW[idx_params])

    @NLconstraint(model, [t = 2:T, j in idx_params], RW[t, j] == RW[t-1, j] + κ_RW[j] * s[j][t])
    @constraint(model, [j in idx_params], 1e-4 ≤ κ_RW[j])
end

"""
# get_num_harmonic_and_seasonal_period(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}})

Retrieve the number of harmonics and seasonal periods specified in the input seasonality dictionary.

## Arguments
- `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: A dictionary indicating the presence of seasonality. If the seasonal component is defined, the value of the dictionary indicates the seasonal period considered.

## Returns
- `num_harmonic::Vector{Union{Int64, Nothing}}`: A vector containing the number of harmonics for each parameter with a seasonal component; otherwise, it contains nothing.
- `seasonal_period::Vector{Int64}`: A vector containing the seasoanl period for each parameter with a seasonal component; otherwise, it contains nothing.
"""
# To do: Troquei o retorno da função caso não tenha sazo para nothing. Lembrar de trocar isso nos if's de outras funções.
function get_num_harmonic_and_seasonal_period(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}})

    num_params = length(seasonality)

    seasonal_period = Vector{Union{Int64, Nothing}}()
    num_harmonic    = Vector{Union{Int64, Nothing}}()

    for i in 1:num_params
        if typeof(seasonality[i]) == Int64
            push!(seasonal_period, seasonality[i])
            push!(num_harmonic, Int64(floor(seasonal_period[i] / 2)))
        else
            push!(seasonal_period, nothing)
            push!(num_harmonic, nothing)
    end
    
    return num_harmonic, seasonal_period
end

"""
# add_trigonometric_seasonality!(model::Ml, s::Vector{Fl}, T::Int64, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, stochastic::Bool=true) where {Ml, Fl}

Incorporates trigonometric seasonality into the specified model, considering the seasonal periods defined in the `seasonality` dictionary.

## Arguments
- `model::Ml`: The model to which trigonometric seasonality will be added.
- `s::Vector{Fl}`: A vector containing the scaled score expressions.
- `T::Int64`: The length of the time series.
- `seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}`: A dictionary indicating the presence of seasonality. If the seasonal component is defined, the value of the dictionary indicates the seasonal period considered.
- `stochastic::Bool=true`: A boolean indicating whether the trigonometric seasonality is stochastic. Default is `true`.

## Returns
- Modifies the input model by adding trigonometric seasonality.
"""
function add_trigonometric_seasonality!(model::Ml, s::Vector{Fl}, T::Int64, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, stochastic::Bool=true) where {Ml, Fl}
    
    num_harmonic, seasonal_period = UnobservedComponentsGAS.get_num_harmonic_and_seasonal_period(seasonality)

    idx_params = findall(i -> i != false, seasonality) # Time-varying parameters with the seasonality dynamic

    unique_num_harmonic = unique(num_harmonic)[minimum(idx_params)]
    #@variable(model, S[1:T, idx_params])
    # @variable(model, κ_S[idx_params])
    # @constraint(model, [i in idx_params], 1e-4 ≤ κ_S[i])

    if stochastic
        @variable(model, κ_S[idx_params])
        @constraint(model, [i in idx_params], 1e-4 ≤ κ_S[i])

        @variable(model, γ[1:unique_num_harmonic, 1:T, idx_params])
        @variable(model, γ_star[1:unique_num_harmonic, 1:T, idx_params])

        @NLconstraint(model, [i = 1:unique_num_harmonic, t = 2:T, j in idx_params], γ[i, t, j] == γ[i, t-1, j] * cos(2*π*i / seasonal_period[j]) + 
                                                                                    γ_star[i,t-1, j]*sin(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t])
        @NLconstraint(model, [i = 1:unique_num_harmonic, t = 2:T, j in idx_params], γ_star[i, t, j] == -γ[i, t-1, j] * sin(2*π*i / seasonal_period[j]) + 
                                                                                    γ_star[i,t-1, j]*cos(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t])

        #@NLconstraint(model, [t = 2:T, j in idx_params], S[t, j] == sum(γ[i, t, j]  for i in 1:unique_num_harmonic))
        @expression(model, S[t = 1:T, j in idx_params], sum(γ[i, t, j]  for i in 1:unique_num_harmonic))
    else

        @variable(model, γ[1:unique_num_harmonic, idx_params])
        @variable(model, γ_star[1:unique_num_harmonic, idx_params])

        # @NLconstraint(model, [t = 2:T, j in idx_params], S[t, j] == sum(γ[i, j]*cos(2 * π * i * t/seasonal_period[j]) + 
        #                                     γ_star[i, j] * sin(2 * π * i* t/seasonal_period[j])  for i in 1:unique_num_harmonic))

        @expression(model, S[t = 1:T, j in idx_params], sum(γ[i, j]*cos(2 * π * i * t/seasonal_period[j]) + 
                                            γ_star[i, j] * sin(2 * π * i* t/seasonal_period[j]) for i in 1:unique_num_harmonic))

    end

end

"""
# include_components!(model::Ml, s::Vector{Fl}, gas_model::GASModel, T::Int64) where {Ml, Fl}

Incorporates various components into the specified model based on the configurations provided in the `GASModel`.

## Arguments
- `model::Ml`: The model to which components will be added.
- `s::Vector{Fl}`: A vector containing the scaled score expressions.
- `gas_model::GASModel`: The GAS model containing configurations for different components.
- `T::Int64`: The length of the time series.

## Returns
- Modifies the input model by adding components such as random walk, random walk slope, autoregressive (AR), and trigonometric seasonality based on the configurations provided in the `GASModel`.
"""
function include_components!(model::Ml, s::Vector{Fl}, gas_model::GASModel, T::Int64) where {Ml, Fl}

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    if has_random_walk(random_walk)
        add_random_walk!(model, s, T, random_walk)
    end

    if has_random_walk_slope(random_walk_slope)
        add_random_walk_slope!(model, s, T, random_walk_slope)
    end

    if has_AR(ar)
        add_AR!(model, s, T, ar)
    end

    if has_seasonality(seasonality)
        add_trigonometric_seasonality!(model, s, T, seasonality, stochastic)
    end
end

"""
# include_component_in_dynamic(model::Ml, component::Symbol, has_component::Bool, t::Int64, idx_param::Int64) where Ml

Returns the expression of a specific component in the dynamic equation of the model if the component is present, otherwise returns 0.

## Arguments
- `model::Ml`: The model containing the dynamic equations.
- `component::Symbol`: The symbol representing the component in the dynamic equation.
- `has_component::Bool`: A boolean indicating whether the component is present in the model.
- `t::Int64`: The time index for which the value of the component is required.
- `idx_param::Int64`: The index of the parameter for which the value of the component is required.

## Returns
- Returns the expression of the specified component in the dynamic equation of the model at the given time index and parameter index, or 0 if the component is not present in the model.
"""
function include_component_in_dynamic(model::Ml, component::Symbol, has_component::Bool, t::Int64, idx_param::Int64) where Ml

    if has_component
        return model[component][t, idx_param]
    else
        return 0
    end
end

"""
# include_explanatories_in_dynamic(model::Ml, X::Union{Missing, Matrix{Float64}}, has_explanatories::Bool, t::Int64, idx_param::Int64) where Ml

Returns the expression of explanatory variables' effect in the dynamic equation of the model if explanatory variables are present, otherwise returns 0.

## Arguments
- `model::Ml`: The model containing the dynamic equations.
- `X::Union{Missing, Matrix{Float64}}`: The matrix of explanatory variables.
- `has_explanatories::Bool`: A boolean indicating whether explanatory variables are present in the model.
- `t::Int64`: The time index for which the value of explanatory variables is required.
- `idx_param::Int64`: The index of the parameter for which the value of explanatory variables is required.

## Returns
- Returns the expression of the explanatory variables' effect in the dynamic equation of the model at the given time index and parameter index, or 0 if explanatory variables are not present in the model.
"""
function include_explanatories_in_dynamic(model::Ml, X::Union{Missing, Matrix{Float64}}, has_explanatories::Bool, t::Int64, idx_param::Int64) where Ml

    if has_explanatories
        return X[t, :]' * model[:β][:, idx_param]
    else
        return 0
    end
end


