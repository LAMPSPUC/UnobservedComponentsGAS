"""
## has_random_walk(level::Vector{String})

Checks if there is any "random walk" component in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.

### Returns
- `true` if there is at least one "random walk" component in the provided level information, `false` otherwise.
"""
has_random_walk(level::Vector{String}) = any(level .== "random walk")

"""
## has_random_walk_slope(level::Vector{String})

Checks if there is any "random walk slope" component in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.

### Returns
- `true` if there is at least one "random walk slope" component in the provided level information, `false` otherwise.
"""
has_random_walk_slope(level::Vector{String}) = any(level .== "random walk slope")

"""
## has_ar1_level(level::Vector{String})

Checks if there is any "ar(1)" component in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.

### Returns
- `true` if there is at least one "ar(1)" component in the provided level information, `false` otherwise.

"""
has_ar1_level(level::Vector{String}) = any(level .== "ar(1)")

"""
## has_level(level::Vector{String})

Checks if there is any level component present in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.

### Returns
- `true` if there is at least one level component present in the provided level information.
- `false` if the level information is empty or contains only empty strings.
"""
has_level(level::Vector{String}) = any(.!isempty.(level))

"""
## has_seasonality(seasonality::Union{String, Vector{String}})

Checks if there is any seasonality component present in the provided seasonality information.

### Arguments
- `seasonality::Union{String, Vector{String}}`: Seasonality information indicating the presence of seasonal components. It can be either a single string or a vector of strings, where each element represents a seasonal component.

### Returns
- `true` if there is at least one seasonal component present in the provided seasonality information.
- `false` if the seasonality information is empty or contains only empty strings.
"""
has_seasonality(seasonality::Union{String, Vector{String}}) = any(.!isempty.(seasonality))

"""
## has_AR(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}})

Checks if there is any AutoRegressive (AR) component present in the provided AR information.

### Arguments
- `ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}`: AR information indicating the presence of AR components. It can be of various types including integers, vectors of integers, vectors of missing values, vectors of union types (combinations of integers and missing values), missing values, vectors of vectors of integers, or vectors of vectors of union types (combinations of vectors of integers and missing values).

### Returns
- `true` if there is at least one AutoRegressive (AR) component present in the provided AR information.
- `false` if the AR information is empty or contains only missing values.
"""
has_AR(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, 
                Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}) = !all(ismissing.(ar))


"""
## has_random_walk(level::Vector{String}, param::Int64)

Checks if there is a "random walk" component at the specified index in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.
- `param::Int64`: The index indicating the position in the `level` vector to check for the presence of a "random walk" component.

### Returns
- `true` if the level component at the specified index is "random walk", `false` otherwise.
"""
has_random_walk(level::Vector{String}, param::Int64) = level[param]== "random walk"

"""
## has_random_walk_slope(level::Vector{String}, param::Int64)

Checks if there is a "random walk slope" component at the specified index in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.
- `param::Int64`: The index indicating the position in the `level` vector to check for the presence of a "random walk slope" component.

### Returns
- `true` if the level component at the specified index is "random walk slope", `false` otherwise.
"""
has_random_walk_slope(level::Vector{String}, param::Int64) = level[param]== "random walk slope"


"""
## has_ar1_level(level::Vector{String}, param::Int64)

Checks if there is an "ar(1)" component at the specified index in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.
- `param::Int64`: The index indicating the position in the `level` vector to check for the presence of an "ar(1)" component.

### Returns
- `true` if the level component at the specified index is "ar(1)", `false` otherwise.
"""
has_ar1_level(level::Vector{String}, param::Int64) = level[param] == "ar(1)"


"""
## has_level(level::Vector{String}, param::Int64)

Checks if there is any level component at the specified index in the provided level information.

### Arguments
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.
- `param::Int64`: The index indicating the position in the `level` vector to check for the presence of a level component.

### Returns
- `true` if there is a level component present at the specified index, `false` otherwise.

"""
has_level(level::Vector{String}, param::Int64) = !isempty(level[param])

"""
## has_seasonality(seasonality::Union{String, Vector{String}}, param::Int64)

Checks if there is any seasonality component at the specified index in the provided seasonality information.

### Arguments
- `seasonality::Union{String, Vector{String}}`: Seasonality information indicating the presence of seasonal components. It can be either a single string or a vector of strings, where each element represents a seasonal component.
- `param::Int64`: The index indicating the position in the `seasonality` information to check for the presence of a seasonality component.

### Returns
- `true` if there is a seasonality component present at the specified index, `false` otherwise.
"""
has_seasonality(seasonality::Union{String, Vector{String}}, param::Int64) = !isempty(seasonality[param]) 

"""
## has_AR(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}, param::Int64)

Checks if there is any AutoRegressive (AR) component at the specified index in the provided AR information.

### Arguments
- `ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}`: AR information indicating the presence of AR components. It can be of various types including integers, vectors of integers, vectors of missing values, vectors of union types (combinations of integers and missing values), missing values, vectors of vectors of integers, or vectors of vectors of union types (combinations of vectors of integers and missing values).
- `param::Int64`: The index indicating the position in the `ar` information to check for the presence of an AR component.

### Returns
- `true` if there is an AutoRegressive (AR) component present at the specified index, `false` otherwise.
"""
has_AR(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, 
       Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}, param::Int64) = !all(ismissing.(ar[param]))


"""
## get_AR_order(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}})

Constructs the AutoRegressive (AR) order based on the provided AR information.

### Arguments
- `ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}`: AR information indicating the AR orders. It can be of various types including integers, vectors of integers, vectors of missing values, vectors of union types (combinations of integers and missing values), missing values, vectors of vectors of integers, or vectors of vectors of union types (combinations of vectors of integers and missing values).

### Returns
- An array representing the AR order. Each element of the array is either a vector of integers representing the AR order or an empty vector if there is no AR component at that index.

### Note
- The function iterates over each element of the AR information and constructs the AR order based on its type and value.
- If the AR information at an index is missing or equals 0, the corresponding element in the AR order array will be `[nothing]`.
- If the AR information at an index is an integer, the corresponding element in the AR order array will be a vector of integers from 1 to the AR value.
- If the AR information at an index is a vector of integers, the corresponding element in the AR order array will be the same vector of integers.

"""
function get_AR_order(ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}})
    
    num_params = length(ar)
    order      = Union{Vector{Int64}, Vector{Nothing}}[]

    for i in 1:num_params

        if ismissing(ar[i]) || ar[i] == 0
            push!(order, [nothing])
        elseif typeof(ar[i]) == Int64
            push!(order, Int64.(collect(1:ar[i])))
        elseif typeof(ar[i]) == Vector{Int64}
            push!(order, Int64.(ar[i]))
        end
    end
    return order
end

"""
## add_AR!(model::Ml, s::Vector{Fl}, T::Int64, ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}) where {Ml, Fl}

Adds AutoRegressive (AR) components to the optimization model.

### Arguments
- `model::Ml`: The optimization model to which the AR components will be added.
- `s::Vector{Fl}`: A vector of time series data.
- `T::Int64`: The length of the time series data.
- `ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}}`: AR information indicating the AR orders. It can be of various types including integers, vectors of integers, vectors of missing values, vectors of union types (combinations of integers and missing values), missing values, vectors of vectors of integers, or vectors of vectors of union types (combinations of vectors of integers and missing values).

### Returns
- Modifies the provided optimization model by adding AR components.
"""
function add_AR!(model::Ml, s::Vector{Fl}, T::Int64, ar::Union{Int64, Vector{Int64}, Vector{Missing}, Vector{Union{Int64, Missing}}, Missing, Vector{Vector{Int64}}, Vector{Union{Vector{Int64}, Missing}}};
                κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2) where {Ml, Fl}

    idx_params = findall(i -> !ismissing(i), ar) # Time-varying parameters with autoregressive dynamic
    order      = get_AR_order(ar)

    max_order     = maximum(filter(x -> !isnothing(x), vcat(order...))) # Maximum lag in the model
    unique_orders = filter(x -> !isnothing(x), unique(vcat(order...)))#[findall(i -> i != 0.0, vcat(order...))]
    
    @variable(model, AR[1:T, idx_params])
    @variable(model, ϕ[1:max_order, idx_params])
    @variable(model, κ_AR[idx_params])

    @constraint(model, [i in idx_params], κ_min ≤ κ_AR[i] ≤ κ_max)

    # Revisar essas restrições com o Alves !!
    for i in unique_orders
        for j in idx_params
            if i ∉ order[j]
                #@constraint(model, ϕ[i, idx_params[j]] == 0)
                JuMP.fix(model[:ϕ][i, j], 0.0)
            end
        end
    end

    @constraint(model, [t = (max_order + 1):T, j in idx_params], AR[t, j] == sum(ϕ[p, j] * AR[t - p, j] for p in unique_orders) + κ_AR[j] * s[j][t])
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
function add_random_walk_slope!(model::Ml, s::Vector{Fl}, T::Int64, random_walk_slope::Dict{Int64, Bool};
                                κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2) where {Ml, Fl}
    
    idx_params = findall(i -> i == true, random_walk_slope) #Time-varying parameters with the random walk with slope dynamic

    @variable(model, RWS[1:T, idx_params])
    @variable(model, b[1:T, idx_params])
    @variable(model, κ_RWS[idx_params])
    @variable(model, κ_b[idx_params])

    @constraint(model, [t = 2:T, j in idx_params], b[t, j]   == b[t - 1, j] + κ_b[j] * s[j][t])
    @constraint(model, [t = 2:T, j in idx_params], RWS[t, j] == RWS[t - 1, j] + b[t - 1, j] + κ_RWS[j] * s[j][t])
    @constraint(model, [j in idx_params], κ_min ≤ κ_RWS[j] ≤ κ_max)
    @constraint(model, [j in idx_params], κ_min ≤ κ_b[j] ≤ κ_min)
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
function add_random_walk!(model::Ml, s::Vector{Fl}, T::Int64, random_walk::Dict{Int64, Bool};
                        κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2) where {Ml, Fl}

    idx_params = findall(i -> i == true, random_walk) # Time-varying parameters with the random walk dynamic

    @variable(model, RW[1:T, idx_params])
    @variable(model, κ_RW[idx_params])

    @constraint(model, [t = 2:T, j in idx_params], RW[t, j] == RW[t-1, j] + κ_RW[j] * s[j][t])
    @constraint(model, [j in idx_params], κ_min ≤ κ_RW[j] ≤ κ_max)
end

"""
## add_ar1!(model::Ml, s::Vector{Fl}, T::Int64, ar1::Dict{Int64, Bool}) where {Fl, Ml}

Adds AutoRegressive (AR(1)) components to the optimization model.

### Arguments
- `model::Ml`: The optimization model to which the AR(1) components will be added.
- `s::Vector{Fl}`: A vector of time series data.
- `T::Int64`: The length of the time series data.
- `ar1::Dict{Int64, Bool}`: A dictionary indicating the presence of AR(1) components, where the keys represent indices and the values represent the presence of an AR(1) component at each index.

### Returns
- Modifies the provided optimization model by adding AR(1) components.
"""
function add_ar1!(model::Ml, s::Vector{Fl}, T::Int64, ar1::Dict{Int64, Bool};
                κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2)  where {Fl, Ml}

    idx_params = findall(i -> i == true, ar1)

    @variable(model, AR1_LEVEL[1:T, idx_params])
    @variable(model, ϕ_AR1_LEVEL[idx_params])
    @variable(model, κ_AR1_LEVEL[idx_params])
    @variable(model, ω_AR1_LEVEL[idx_params])

    @constraint(model, [i in idx_params], κ_min ≤ κ_AR1_LEVEL[i] ≤ κ_max)
    @constraint(model, [i in idx_params], -0.95 <= ϕ_AR1_LEVEL[i] <= 0.95)

    @constraint(model, [t = 2:T, j in idx_params], AR1_LEVEL[t, j] == ω_AR1_LEVEL[j] + ϕ_AR1_LEVEL[j] * AR1_LEVEL[t-1, j] + κ_AR1_LEVEL[j] * s[j][t])

end

"""
## add_level!(model::Ml, s::Vector{Fl}, T::Int64, level::Vector{String}) where {Fl, Ml}

Adds level components to the optimization model based on the provided level information.

### Arguments
- `model::Ml`: The optimization model to which the level components will be added.
- `s::Vector{Fl}`: A vector of time series data.
- `T::Int64`: The length of the time series data.
- `level::Vector{String}`: A vector indicating the presence of level components, where each element represents a level component.

### Returns
- Modifies the provided optimization model by adding level components.
"""
function add_level!(model::Ml, s::Vector{Fl}, T::Int64, level::Vector{String};
                    κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2) where {Fl, Ml}

    if "random walk" ∈ level 
        random_walk = Dict{Int64, Bool}()
        for i in 1:length(level)
            level[i] == "random walk" ? random_walk[i] = true : random_walk[i] = false
        end
        add_random_walk!(model, s, T, random_walk; κ_min = κ_min, κ_max = κ_max)
    end
    
    if "random walk slope" ∈ level 
        random_walk_slope = Dict{Int64, Bool}()
        for i in 1:length(level)
            level[i] == "random walk slope" ? random_walk_slope[i] = true : random_walk_slope[i] = false
        end
        add_random_walk_slope!(model, s, T, random_walk_slope; κ_min = κ_min, κ_max = κ_max)
    end
    
    if "ar(1)" ∈ level 
        ar1 = Dict{Int64, Bool}()
        for i in 1:length(level)
            level[i] == "ar(1)" ? ar1[i] = true : ar1[i] = false
        end
        add_ar1!(model, s, T, ar1; κ_min = κ_min, κ_max = κ_max)
    end
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
function get_num_harmonic_and_seasonal_period(seasonality::Dict{Int64, Union{Bool, Int64}}; 
                                                fix_num_harmonic::Vector{U} = [missing, missing]) where{U}

    num_params = length(seasonality)

    seasonal_period = Vector{Union{Int64, Nothing}}()
    num_harmonic    = Vector{Union{Int64, Nothing}}()

    for i in 1:num_params
        if typeof(seasonality[i]) == Int64

            push!(seasonal_period, seasonality[i])
            max_num_harmonic = Int64(floor(seasonal_period[i] / 2))

            if ismissing(fix_num_harmonic[i])
                push!(num_harmonic, max_num_harmonic)
            else
                if fix_num_harmonic[i] > max_num_harmonic
                    @warn "Num harmonics should be less or equal than $max_num_harmonic. Fixing num harmonics to $max_num_harmonic."
                    fix_num_harmonic[i] = max_num_harmonic
                end
                push!(num_harmonic, fix_num_harmonic[i])
            end
        else
            push!(seasonal_period, nothing)
            push!(num_harmonic, nothing)
        end
    end
    
    return num_harmonic, seasonal_period
end

"""
## add_trigonometric_seasonality!(model::Ml, s::Vector{Fl}, T::Int64, seasonality::Vector{String}) where {Ml, Fl}

Adds trigonometric seasonality components to the optimization model based on the provided seasonality information.

### Arguments
- `model::Ml`: The optimization model to which the trigonometric seasonality components will be added.
- `s::Vector{Fl}`: A vector of time series data.
- `T::Int64`: The length of the time series data.
- `seasonality::Vector{String}`: A vector indicating the presence of seasonal components, where each element represents a seasonal component.

### Returns
- Modifies the provided optimization model by adding trigonometric seasonality components.
"""
function add_trigonometric_seasonality!(model::Ml, s::Vector{Fl}, T::Int64, seasonality::Vector{String};
                                        κ_min::Union{Float64, Int64} = 1e-5, κ_max_s::Union{Float64, Int64} = 1,
                                        fix_num_harmonic::Vector{U} = [missing, missing]) where {Ml, Fl, U}
    
    seasonality_dict, stochastic, stochastic_params = get_seasonality_dict_and_stochastic(seasonality)
    
    num_harmonic, seasonal_period = get_num_harmonic_and_seasonal_period(seasonality_dict; fix_num_harmonic = fix_num_harmonic)
    idx_params = sort(findall(i -> i != false, seasonality_dict)) # Time-varying parameters with the seasonality dynamic
    idx_params_deterministic = idx_params[.!stochastic_params[idx_params]]
    idx_params_stochastic    = idx_params[stochastic_params[idx_params]]
    
    unique_num_harmonic = unique(num_harmonic)[minimum(idx_params)]

    S_aux = Matrix(undef, T, length(seasonality))

    if !isempty(idx_params_stochastic)
        @variable(model, κ_S[idx_params_stochastic])
        @constraint(model, [i in idx_params_stochastic], κ_min  ≤ κ_S[i] ≤ κ_max_s)
        # @constraint(model, [i in idx_params_stochastic], 0.0 ≤ κ_S[i] ≤ 0.0)

        @variable(model, γ_sto[1:unique_num_harmonic, 1:T, idx_params_stochastic])
        @variable(model, γ_star_sto[1:unique_num_harmonic, 1:T, idx_params_stochastic])

        @constraint(model, [i = 1:unique_num_harmonic, t = 2:T, j in idx_params_stochastic], γ_sto[i, t, j] == γ_sto[i, t-1, j] * cos(2*π*i / seasonal_period[j]) + 
                                                                                    γ_star_sto[i,t-1, j]*sin(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t])
        @constraint(model, [i = 1:unique_num_harmonic, t = 2:T, j in idx_params_stochastic], γ_star_sto[i, t, j] == -γ_sto[i, t-1, j] * sin(2*π*i / seasonal_period[j]) + 
                                                                                    γ_star_sto[i,t-1, j]*cos(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t])

        # Define the decision variables only for the initial time step
        # @variable(model, γ_sto_ini[1:unique_num_harmonic, idx_params_stochastic])
        # @variable(model, γ_star_sto_ini[1:unique_num_harmonic, idx_params_stochastic])
        # γ_sto = Array{NonlinearExpr}(undef, unique_num_harmonic, T, length(idx_params_stochastic))
        # γ_star_sto = Array{NonlinearExpr}(undef, unique_num_harmonic, T, length(idx_params_stochastic))
        # println("Criando expression")
        # # Create JuMP expressions to calculate γ_sto and γ_star_sto over time
        # for i in 1:unique_num_harmonic, t in 1:T, j in idx_params_stochastic
        #     if t == 1
        #         # For t == 1, use the initial values
        #         γ_sto[i, t, j] = γ_sto_ini[i, j]
        #         γ_star_sto[i, t, j] = γ_star_sto_ini[i, j]
        #     else
        #         # For t > 1, use the recurrence relations
        #         γ_sto[i, t, j] = γ_sto[i, t-1, j] * cos(2*π*i / seasonal_period[j]) + 
        #                             γ_star_sto[i, t-1, j] * sin(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t]
        #         γ_star_sto[i, t, j] = -γ_sto[i, t-1, j] * sin(2*π*i / seasonal_period[j]) + 
        #                                     γ_star_sto[i, t-1, j] * cos(2*π*i / seasonal_period[j]) + κ_S[j] * s[j][t]
        #     end
        # end

        # @expression(model, γ_sto, γ_sto)
        # @expression(model, γ_star_sto, γ_star_sto)
        # println("Fim criação expression")
        for j in idx_params_stochastic  
            for t in 1:T
                S_aux[t, j] = sum(γ_sto[i, t, j]  for i in 1:unique_num_harmonic)
            end
        end
        #@expression(model, S[t = 1:T, j in idx_params], sum(γ[i, t, j]  for i in 1:unique_num_harmonic))
    end

    if !isempty(idx_params_deterministic)

        @variable(model, γ_det[1:unique_num_harmonic, idx_params_deterministic])
        @variable(model, γ_star_det[1:unique_num_harmonic, idx_params_deterministic])

        for j in idx_params_deterministic  
            for t in 1:T
                S_aux[t, j] = sum(γ_det[i, j]*cos(2 * π * i * t/seasonal_period[j]) + 
                                            γ_star_det[i, j] * sin(2 * π * i* t/seasonal_period[j]) for i in 1:unique_num_harmonic)
            end
        end
        #@expression(model, S[t = 1:T, j in idx_params], sum(γ[i, j]*cos(2 * π * i * t/seasonal_period[j]) + 
                #                            γ_star[i, j] * sin(2 * π * i* t/seasonal_period[j]) for i in 1:unique_num_harmonic))
    end

    @expression(model, S[t=1:T, j in idx_params], S_aux[t, j])

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
function include_components!(model::Ml, s::Vector{Fl}, gas_model::GASModel, T::Int64;
                            κ_min::Union{Float64, Int64} = 1e-5, κ_max::Union{Float64, Int64} = 2,
                            κ_max_s::Union{Float64, Int64} = 1, fix_num_harmonic::Vector{U} = [missing, missing]) where {Ml, Fl, U}

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model
    
    if has_level(level)
        add_level!(model, s, T, level; κ_min = κ_min, κ_max = κ_max)
    end
    
    if has_AR(ar)
        add_AR!(model, s, T, ar; κ_min = κ_min, κ_max = κ_max)
    end

    if has_seasonality(seasonality)
        add_trigonometric_seasonality!(model, s, T, seasonality; κ_min = κ_min, κ_max_s = κ_max_s, fix_num_harmonic = fix_num_harmonic)
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


