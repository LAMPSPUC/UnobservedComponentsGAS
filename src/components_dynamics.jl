has_random_walk(random_walk::Dict{Int64, Bool})                                      = any(values(random_walk))
has_random_walk_slope(random_walk_slope::Dict{Int64, Bool})                          = any(values(random_walk_slope))
has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}})           = !all(isequal.(typeof.(values(seasonality)), Bool))
has_AR(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}) = !all(isequal.(typeof.(values(ar)), Bool))

has_random_walk(random_walk::Dict{Int64, Bool}, param::Int64)                                      = random_walk[param]
has_random_walk_slope(random_walk_slope::Dict{Int64, Bool}, param::Int64)                          = random_walk_slope[param]
has_seasonality(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}, param::Int64)           = seasonality[param] != false 
has_AR(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}, param::Int64) = typeof(ar[param]) == Bool || ar[param] == 0 ? false : true

"
Returns a vector with the autoregressive lags to be considered in the model, for each time-varying parameter.
"
function get_AR_order(ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}})
    
    num_params = length(ar)
    order      = Vector{Int64}[]

    for i in 1:num_params
        if typeof(ar[i]) == Int64
            push!(order, Int64.(collect(1:ar[i])))
        elseif typeof(ar[i]) == Vector{Int64}
            push!(order, Int64.(ar[i]))
        else
            push!(order, [0])
        end
    end
    return order
end

"
Add the variables and constraints of the autoregressive dynamic to a JuMP model.
"
function add_AR!(model::Ml, s::Vector{Fl}, T::Int64, ar::Union{Dict{Int64, Int64}, Dict{Int64, Vector{Int64}}, Dict{Int64, Bool}, Dict{Int64, Any}}) where {Ml, Fl}

    idx_params = findall(i -> i != 0, ar) # Time-varying parameters with autoregressive dynamic
    order      = get_AR_order(ar)

    max_order     = maximum(vcat(order...)) # Maximum lag in the model
    unique_orders = filter(x -> x != 0.0, unique(vcat(order...)))#[findall(i -> i != 0.0, vcat(order...))]
    
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

"
Add the variables and constraints of the random walk with slope dynamic to a JuMP model.
"
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

"
Add the variables and constraints of the random walk dynamic to a JuMP model.
"
function add_random_walk!(model::Ml, s::Vector{Fl}, T::Int64, random_walk::Dict{Int64, Bool}) where {Ml, Fl}

    idx_params = findall(i -> i == true, random_walk) # Time-varying parameters with the random walk dynamic

    @variable(model, RW[1:T, idx_params])
    @variable(model, κ_RW[idx_params])

    @NLconstraint(model, [t = 2:T, j in idx_params], RW[t, j] == RW[t-1, j] + κ_RW[j] * s[j][t])
    @constraint(model, [j in idx_params], 1e-4 ≤ κ_RW[j])
end

"
Returns the number of harmonic and seasonal periods for each time-varying parameter.
"
function get_num_harmonic_and_seasonal_period(seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}})

    num_params = length(seasonality)

    seasonal_period = Int64[]
    num_harmonic    = Int64[]

    for i in 1:num_params
        if typeof(seasonality[i]) == Int64
            push!(seasonal_period, seasonality[i])
            push!(num_harmonic, Int64(floor(seasonal_period[i] / 2)))
        end
    end
    
    return num_harmonic, seasonal_period
end

"
Add the variables and constraints of the trigonometric seasonality dynamic to a JuMP model.
"
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

"
Add all the components to the JuMP model.
"
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

"Include a given component to the parameters dynamic if its necessary, otherwise, return 0.
Used in the construction of the JuMP model.
"
function include_component_in_dynamic(model::Ml, component::Symbol, has_component::Bool, t::Int64, idx_param::Int64) where Ml

    if has_component
        return model[component][t, idx_param]
    else
        return 0
    end
end

"Include explanatories to the parameters dynamic if its necessary, otherwise, return 0"
function include_explanatories_in_dynamic(model::Ml, X::Union{Missing, Matrix{Float64}}, has_explanatories::Bool, t::Int64, idx_param::Int64) where Ml

    if has_explanatories
        return X[t, :]' * model[:β][:, idx_param]
    else
        return 0
    end
end


