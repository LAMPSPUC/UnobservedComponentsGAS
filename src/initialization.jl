# Fit and return the predictive state of a StateSpaceModel
function fit_and_get_preditive_state(model::M) where M
    StateSpaceModels.fit!(model)
    return StateSpaceModels.get_predictive_state(model)
end

# Case without explanatory
function define_state_space_model(y::Vector{Float64}, has_level::Bool, has_slope::Bool, 
                                has_seasonality::Bool, seasonal_period::Union{Missing, Int64}, stochastic::Bool)

    T                   = length(y)
    initial_level       = zeros(T)
    initial_slope       = zeros(T)
    initial_seasonality = zeros(T)
    initial_γ           = zeros(1)
    initial_γ_star      = zeros(1)

    if has_seasonality
        if !has_level && !has_slope
            println("Entrou aqui")
            ss_model = SeasonalNaive(y, seasonal_period)
            StateSpaceModels.fit!(ss_model)
            initial_seasonality = ss_model.y .- vcat(zeros(seasonal_period), ss_model.residuals)
            res = ss_model.residuals
            initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, seasonal_period, stochastic)
        else
            #Basic structural
            ss_model                  = BasicStructural(y, seasonal_period)
            pred_state                = fit_and_get_preditive_state(ss_model)
            for t in 1:T
                initial_seasonality[t] = -sum(pred_state[t+1, end-(seasonal_period-2):end])
            end
            initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, seasonal_period, stochastic)

            if has_level && has_slope
                initial_level = pred_state[2:end,1]
                initial_slope = pred_state[2:end,2]
            elseif has_level && !has_slope
                initial_level = pred_state[2:end,1] + pred_state[2:end,2]
            end
            res = StateSpaceModels.get_innovations(ss_model)[:, 1]
        end
    else
        if has_level && has_slope
            #Local Linear Trend
            ss_model   = LocalLinearTrend(y)
            pred_state = fit_and_get_preditive_state(ss_model)
            initial_level = pred_state[2:end,1]
            initial_slope = pred_state[2:end,2]
        elseif has_level && !has_slope
            # Local Level
            ss_model   = LocalLevel(y)
            pred_state = fit_and_get_preditive_state(ss_model)
            initial_level = pred_state[2:end,1]
        end
        res = StateSpaceModels.get_innovations(ss_model)[:, 1]
    end

    if length(res) < T
        res = vcat(rand(res, T - length(res)), res)
    end
    
    return Dict("level" => initial_level,"slope" => initial_slope,"seasonality" => initial_seasonality,
            "γ" => initial_γ,"γ_star" => initial_γ_star,"explanatory" => missing,"res" => res)
end

# Case with explanatory
function define_state_space_model(y::Vector{Float64}, X::Union{Matrix{Float64}, Missing}, has_level::Bool, has_slope::Bool, 
                                has_seasonality::Bool, seasonal_period::Union{Missing, Int64}, stochastic::Bool)

    T                   = length(y)
    N                   = size(X, 2)
    initial_level       = zeros(T)
    initial_slope       = zeros(T)
    initial_seasonality = zeros(T)
    initial_γ           = zeros(1)
    initial_γ_star      = zeros(1)
    explanatory_coefs   = zeros(N)
    
    # Be aware that if we select just seasonal as the mean component with explanatories, 
    ## there wont be a initial value for the explanatories coefs (it will be zero)
    if has_seasonality
        if !has_level && !has_slope
            ss_model = SeasonalNaive(y, seasonal_period)
            StateSpaceModels.fit!(ss_model)
            initial_seasonality = ss_model.y .- vcat(zeros(seasonal_period), ss_model.residuals)
            explanatory_coefs   = zeros(N)
            res = ss_model.residuals
            initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, seasonal_period, stochastic)
        else
         #Basic structural
            ss_model   = BasicStructuralExplanatory(y, seasonal_period, X)
            pred_state = fit_and_get_preditive_state(ss_model)
            for t in 1:T
                initial_seasonality[t] = -sum(pred_state[t+1, end-(seasonal_period+N-2):end-N])
            end
            initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, seasonal_period, stochastic)

            if has_level && has_slope
                initial_level = pred_state[2:end,1]
                initial_slope = pred_state[2:end,2]
            elseif has_level && !has_slope
                initial_level = pred_state[2:end,1] + pred_state[2:end,2]
            end
            res = StateSpaceModels.get_innovations(ss_model)[:, 1]
        end
    else
        if has_level && has_slope 
            # Since there is no LocalLinearTrendExplanatory ...
            ss_model   = BasicStructuralExplanatory(y, 12, X)
            pred_state = fit_and_get_preditive_state(ss_model)
            initial_level = pred_state[2:end,1]
            initial_slope = pred_state[2:end,2]
        elseif has_level && !has_slope
            # Local Level
            ss_model   = LocalLevelExplanatory(y, X)
            pred_state = fit_and_get_preditive_state(ss_model)
            initial_level = pred_state[2:end,1]
        end
        res = StateSpaceModels.get_innovations(ss_model)[:, 1]
    end
    
    explanatory_coefs = ss_model.hyperparameters.constrained_values[end-N+1:end]

    if length(res) < T
        res = vcat(rand(res, T - length(res)), res)
    end

    return Dict("level" => initial_level,"slope" => initial_slope,"seasonality" => initial_seasonality,
                "γ" => initial_γ,"γ_star" => initial_γ_star,"explanatory" => explanatory_coefs,"res" => res)
end


function get_initial_values(y::Vector{Float64}, X::Union{Matrix{Float64}, Missing}, has_level::Bool, has_ar1_level::Bool, has_slope::Bool, has_seasonality::Bool, seasonal_period::Union{Missing, Int64}, stochastic::Bool, order::Union{Vector{Int64}, Vector{Nothing}}, max_order::Int64)

    #T = length(y)
    has_explanatories = !ismissing(X) ? true : false

    if has_level || has_slope || has_seasonality

        if has_explanatories
            ss_components = define_state_space_model(y, X, has_level, has_slope, has_seasonality, seasonal_period, stochastic)
        else
            ss_components = define_state_space_model(y, has_level, has_slope, has_seasonality, seasonal_period, stochastic)
        end

        if !isnothing(order[1])
            res = ss_components["res"]
            fit_ar_model, ar_coefs, ar_intercept = fit_AR_model(res, order)

            initial_ar = fit_ar_model
            initial_ϕ  = zeros(max_order)
            for i in eachindex(order)
                initial_ϕ[order[i]] = ar_coefs[order[i]]
            end
        else
            initial_ar = zeros(length(y))
            initial_ϕ  =  zeros(max_order)
        end

        initial_intercept = 0.0 #output.coefs[1]
    else
        fit_ar_model, ar_coefs, ar_intercept = fit_AR_model(y, order)

        initial_ar = fit_ar_model
            initial_ϕ  = zeros(max_order)
            for i in eachindex(order)
                initial_ϕ[order[i]] = ar_coefs[order[i]]
            end

        initial_intercept = ar_intercept
    end

    if has_ar1_level
        initial_ar1_level = ss_components["level"]
        initial_rws       = zeros(length(y))
        initial_slope     = zeros(length(y))
        initial_rw        = zeros(length(y))
    elseif has_slope && has_level
        initial_ar1_level = zeros(length(y))
        initial_rws       = ss_components["level"]
        initial_slope     = ss_components["slope"]
        initial_rw        = zeros(length(y))
    elseif !has_slope && has_level
        initial_ar1_level = zeros(length(y))
        initial_rw        = ss_components["level"]
        initial_rws       = zeros(length(y))
        initial_slope     = zeros(length(y))
    elseif !has_slope && !has_level
        initial_ar1_level = zeros(length(y))
        initial_rws       = zeros(length(y))
        initial_slope     = zeros(length(y))
        initial_rw        = zeros(length(y))
    end

    if has_seasonality
        initial_seasonality = ss_components["seasonality"]
        initial_γ           = ss_components["γ"]
        initial_γ_star      = ss_components["γ_star"]
    else
        initial_seasonality = zeros(length(y))
        initial_γ           = zeros(1)
        initial_γ_star      = zeros(1)
    end

    selected_explanatories = missing

    initial_values                          = Dict()
    initial_values["intercept"]             = Dict()
    initial_values["intercept"]["values"]   = initial_intercept
    initial_values["rw"]                    = Dict()
    initial_values["rw"]["values"]          = initial_rw
    initial_values["rw"]["κ"]               = 0.02
    initial_values["rws"]                   = Dict()
    initial_values["rws"]["values"]         = initial_rws
    initial_values["rws"]["κ"]              = 0.02
    initial_values["ar1_level"]             = Dict()
    initial_values["ar1_level"]["values"]   = initial_ar1_level
    initial_values["ar1_level"]["κ"]        = 0.02
    initial_values["ar1_level"]["ϕ"]        = 0.7
    initial_values["slope"]                 = Dict()
    initial_values["slope"]["values"]       = initial_slope
    initial_values["slope"]["κ"]            = 0.02
    initial_values["seasonality"]           = Dict()
    initial_values["seasonality"]["values"] = initial_seasonality
    initial_values["seasonality"]["γ"]      = initial_γ
    initial_values["seasonality"]["γ_star"] = initial_γ_star
    initial_values["seasonality"]["κ"]      = 0.02
    initial_values["ar"]                    = Dict()
    initial_values["ar"]["values"]          = initial_ar
    initial_values["ar"]["ϕ"]               = initial_ϕ
    initial_values["ar"]["κ"]               = 0.02

    if has_explanatories
        initial_values["explanatories"] = ss_components["explanatory"]
    end

    return initial_values
end
  
function create_output_initialization(y::Vector{Fl}, X::Union{Matrix{Fl}, Missing}, gas_model::GASModel) where {Fl}

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model
    
    dist_code               = get_dist_code(dist)
    num_params              = get_num_params(dist)
    idx_time_varying_params = get_idxs_time_varying_params(time_varying_params)
    idx_fixed_params        = setdiff(1:num_params, idx_time_varying_params)
    T                       = length(y)
    order                   = get_AR_order(ar)
    max_order               = has_AR(ar) ? max_order = maximum(filter(x -> !isnothing(x), vcat(order...))) : 0

    seasonality_dict, stochastic = get_seasonality_dict_and_stochastic(seasonality)
    initial_params = get_initial_params(y, time_varying_params, dist, seasonality_dict)

    initial_values = Vector{Any}(undef, maximum(idx_time_varying_params))

    has_level       = zeros(Bool, num_params)
    has_level_ar1   = zeros(Bool, num_params)
    has_slope       = zeros(Bool, num_params)
    has_seasonal    = zeros(Bool, num_params)
    seasonal_period = zeros(Int64, num_params)
    
    for i in idx_time_varying_params

        #checking which components will be consider 
        if has_random_walk_slope(level, i)
            has_level[i]     = true
            has_slope[i]     = true
            has_level_ar1[i] = false
        
        elseif has_random_walk(level, i)
            has_level[i]     = true
            has_slope[i]     = false
            has_level_ar1[i] = false
        
        elseif has_ar1_level(level, i)
            has_level[i]     = true
            has_slope[i]     = false
            has_level_ar1[i] = true

        else #!has_random_walk_slope(level, i) && !has_random_walk(level, i)
            has_level[i] = false
            has_slope[i] = false
            has_level_ar1[i] = true
        end

        has_seasonal[i] = has_seasonality(seasonality, i)
        if has_seasonal[i]
            seasonal_period[i] = get_num_harmonic_and_seasonal_period(seasonality_dict)[2][i]
        end

        X_aux =  i == 1 && !ismissing(X) ? X : missing
        println("Param = $i")
        println("LEVEL ",has_level)
        println("SLOPE ",has_slope)
        println("seasonal ",has_seasonal)
        initial_values[i] = get_initial_values(initial_params[i], X_aux, has_level[i], has_level_ar1[i], has_slope[i], has_seasonal[i], seasonal_period[i], stochastic, order[i], max_order)
        
        # initialize the mean parameter as the sum of the initial values of the components
        # initial_params[i] = zeros(T)
        # for k in ["rw", "rws", "slope", "seasonality", "explanatories", "ar"]
        #     if haskey(initial_values[i], k)
        #         if k != "explanatories"
        #             initial_params[i] += initial_values[i][k]["values"]
        #         else
        #             initial_params[i] += X * initial_values[i][k]
        #         end
        #     end
        # end
    end

    aux_params = Matrix(undef, T, num_params)
    aux_fixed_params = Vector(undef, num_params)
    for i in 1:num_params
       if i in idx_time_varying_params
            aux_params[:, i] = initial_params[i]
       else
            aux_fixed_params[i] = initial_params[i]
       end
    end


    output_initial_values = initial_values[minimum(idx_time_varying_params)]

    output_initial_values["param"] = aux_params[:, idx_time_varying_params]

    if length(idx_time_varying_params) != length(time_varying_params)
        output_initial_values["fixed_param"] = aux_fixed_params[idx_fixed_params]
    end

    if length(idx_time_varying_params) > 1
        for i in eachindex(time_varying_params)
            if i != minimum(idx_time_varying_params)
                if has_ar1_level(level, i)
                    output_initial_values["ar1_level"]["values"] = hcat(output_initial_values["ar1"]["values"], initial_values[i]["ar1"]["values"])
                    output_initial_values["ar1_level"]["κ"] = vcat(output_initial_values["ar1_level"]["κ"], initial_values[i]["ar1_level"]["κ"])
                    output_initial_values["ar1_level"]["ϕ"] = vcat(output_initial_values["ar1_level"]["ϕ"], initial_values[i]["ar1_level"]["ϕ"])
                end

                if has_random_walk(level, i)
                    output_initial_values["rw"]["values"] = hcat(output_initial_values["rw"]["values"], initial_values[i]["rw"]["values"])
                    output_initial_values["rw"]["κ"] = vcat(output_initial_values["rw"]["κ"], initial_values[i]["rw"]["κ"])
                end

                if has_random_walk_slope(level, i)
                    output_initial_values["rws"]["values"] = hcat(output_initial_values["rws"]["values"], initial_values[i]["rws"]["values"])
                    output_initial_values["rws"]["κ"] = vcat(output_initial_values["rws"]["κ"], initial_values[i]["rws"]["κ"])

                    output_initial_values["slope"]["values"] = hcat(output_initial_values["slope"]["values"], initial_values[i]["slope"]["values"])
                    output_initial_values["slope"]["κ"] = vcat(output_initial_values["slope"]["κ"], initial_values[i]["slope"]["κ"])
                end

                if has_seasonality(seasonality, i)
                    output_initial_values["seasonality"]["values"] = hcat(output_initial_values["seasonality"]["values"], initial_values[i]["seasonality"]["values"])
                    if stochastic
                        output_initial_values["seasonality"]["γ"] = cat(output_initial_values["seasonality"]["γ"], initial_values[i]["seasonality"]["γ"], dims = 3)
                        output_initial_values["seasonality"]["γ_star"] = cat(output_initial_values["seasonality"]["γ_star"], initial_values[i]["seasonality"]["γ_star"], dims = 3)
                    else
                        output_initial_values["seasonality"]["γ"] = hcat(output_initial_values["seasonality"]["γ"], initial_values[i]["seasonality"]["γ"])
                        output_initial_values["seasonality"]["γ_star"] = hcat(output_initial_values["seasonality"]["γ_star"], initial_values[i]["seasonality"]["γ_star"])
                    end
                    output_initial_values["seasonality"]["κ"] = vcat(output_initial_values["seasonality"]["κ"], initial_values[i]["seasonality"]["κ"])
                end  
                
                if has_AR(ar, i)
                    output_initial_values["ar"]["ϕ"] = hcat(output_initial_values["ar"]["ϕ"], initial_values[i]["ar"]["ϕ"])
                    output_initial_values["ar"]["κ"] = vcat(output_initial_values["ar"]["κ"], initial_values[i]["ar"]["κ"])
                    output_initial_values["ar"]["values"] = hcat(output_initial_values["ar"]["values"], initial_values[i]["ar"]["values"])
                end
            end
        end
    end

    return convert(Dict{String, Any}, output_initial_values)
end

function create_output_initialization_from_fit(output::Output, gas_model::GASModel)

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    num_params              = get_num_params(dist)
    idx_time_varying_params = get_idxs_time_varying_params(time_varying_params)
    idx_fixed_params        = setdiff(1:num_params, idx_time_varying_params)

    if typeof(dist) == LogNormalDistribution
        fitted_params = convert_to_log_scale(output.fitted_params)
    else
        fitted_params = output.fitted_params
    end
    
    components    = output.components
    T             = length(fitted_params["param_1"])

    order                   = get_AR_order(ar)
    max_order               = has_AR(ar) ? maximum(filter(x -> !isnothing(x), vcat(order...))) : 0

    output_initial_values       = Dict()
    initial_time_varying_params = zeros(length(fitted_params["param_1"]), num_params) 
    initial_fixed_params        = zeros(num_params)

    for i in idx_time_varying_params
        initial_time_varying_params[:, i] = fitted_params["param_$i"]
    end

    for i in idx_fixed_params
        initial_fixed_params[i] = fitted_params["param_$i"][1]
    end

    output_initial_values["param"]       = initial_time_varying_params[:, idx_time_varying_params]
    output_initial_values["fixed_param"] = initial_fixed_params[idx_fixed_params]

    # Getting initial values for the mean parameter
    output_initial_values["rw"]                  = Dict()
    output_initial_values["rws"]                 = Dict()
    output_initial_values["ar1_level"]           = Dict()
    output_initial_values["slope"]               = Dict()
    output_initial_values["seasonality"]         = Dict()
    output_initial_values["ar"]                  = Dict()
    output_initial_values["intercept"]           = Dict()
    output_initial_values["intercept"]["values"] = components["param_1"]["intercept"]

    if has_random_walk(level, 1)
        output_initial_values["rw"]["values"] = components["param_1"]["level"]["value"]
        output_initial_values["rw"]["κ"]        = components["param_1"]["level"]["hyperparameters"]["κ"]
    else
        output_initial_values["rw"]["values"] = zeros(T)
        output_initial_values["rw"]["κ"]      = 0.0
    end

    if has_ar1_level(level, 1)
        output_initial_values["ar1_level"]["values"] = components["param_1"]["level"]["value"]
        output_initial_values["ar1_level"]["κ"]      = components["param_1"]["level"]["hyperparameters"]["κ"]
        output_initial_values["ar1_level"]["ϕ"]      = components["param_1"]["level"]["hyperparameters"]["ϕ"]
    else
        output_initial_values["ar1_level"]["values"] = zeros(T)
        output_initial_values["ar1_level"]["κ"]      = 0.0
        output_initial_values["ar1_level"]["ϕ"]      = 0.0
    end 

    if has_random_walk_slope(level, 1)
        output_initial_values["rws"]["values"]   = components["param_1"]["level"]["value"]
        output_initial_values["rws"]["κ"]        = components["param_1"]["level"]["hyperparameters"]["κ"]
        output_initial_values["slope"]["values"] = components["param_1"]["slope"]["value"]
        output_initial_values["slope"]["κ"]      = components["param_1"]["slope"]["hyperparameters"]["κ"]
    else
        output_initial_values["rws"]["values"]   = zeros(T)
        output_initial_values["rws"]["κ"]        = 0.0
        output_initial_values["slope"]["values"] = zeros(T)
        output_initial_values["slope"]["κ"]      = 0.0
    end

    if has_seasonality(seasonality, 1)
        seasonality_dict, stochastic = get_seasonality_dict_and_stochastic(seasonality)
        output_initial_values["seasonality"]["values"] = components["param_1"]["seasonality"]["value"]
        if stochastic
            output_initial_values["seasonality"]["κ"]      = components["param_1"]["seasonality"]["hyperparameters"]["κ"]
        end
        output_initial_values["seasonality"]["γ"]      = components["param_1"]["seasonality"]["hyperparameters"]["γ"]
        output_initial_values["seasonality"]["γ_star"] = components["param_1"]["seasonality"]["hyperparameters"]["γ_star"]
    else
        output_initial_values["seasonality"]["values"] = zeros(T)
        output_initial_values["seasonality"]["κ"]      = 0.0
        output_initial_values["seasonality"]["γ"]      = 0.0
        output_initial_values["seasonality"]["γ_star"] = 0.0
    end

    if has_AR(ar, 1)
        output_initial_values["ar"]["ϕ"] = components["param_1"]["ar"]["hyperparameters"]["ϕ"]
        output_initial_values["ar"]["κ"] = components["param_1"]["ar"]["hyperparameters"]["κ"]
        output_initial_values["ar"]["values"] = components["param_1"]["ar"]["value"]
    else
        output_initial_values["ar"]["ϕ"] = zeros(max_order)
        output_initial_values["ar"]["values"] = zeros(T)
        output_initial_values["ar"]["κ"] = 0.0
    end

    if haskey(components["param_1"], "explanatories")
        output_initial_values["explanatories"] = components["param_1"]["explanatories"]
    end

    if length(components) > 1
        for i in setdiff(idx_time_varying_params, 1)
            output_initial_values["intercept"]["values"]  = vcat(output_initial_values["intercept"]["values"], components["param_$i"]["intercept"])

            if has_random_walk(level, i)
                output_initial_values["rw"]["values"] = hcat(output_initial_values["rw"]["values"], components["param_$i"]["level"]["value"])
                output_initial_values["rw"]["κ"]      = vcat(output_initial_values["rw"]["κ"] , components["param_$i"]["level"]["hyperparameters"]["κ"])
            end

            if has_random_ar1_level(level, i)
                output_initial_values["ar1_level"]["values"] = hcat(output_initial_values["ar1_level"]["values"], components["param_$i"]["level"]["value"])
                output_initial_values["ar1_level"]["κ"]      = vcat(output_initial_values["ar1_level"]["κ"] , components["param_$i"]["level"]["hyperparameters"]["κ"])
                output_initial_values["ar1_level"]["ϕ"]      = vcat(output_initial_values["ar1_level"]["ϕ"] , components["param_$i"]["level"]["hyperparameters"]["ϕ"])
            end

            if has_random_walk_slope(level, i)
                output_initial_values["rws"]["values"]   = hcat(output_initial_values["rws"]["values"], components["param_$i"]["level"]["value"])
                output_initial_values["rws"]["κ"]        = vcat(output_initial_values["rws"]["κ"], components["param_$i"]["level"]["hyperparameters"]["κ"])
                output_initial_values["slope"]["values"] = hcat(output_initial_values["slope"]["values"], components["param_$i"]["slope"]["value"])
                output_initial_values["slope"]["κ"]      = vcat(output_initial_values["slope"]["κ"], components["param_$i"]["slope"]["hyperparameters"]["κ"])
            end

            if has_seasonality(seasonality, i)
                output_initial_values["seasonality"]["values"] = hcat(output_initial_values["seasonality"]["values"], components["param_$i"]["seasonality"]["value"])
                if stochastic
                    output_initial_values["seasonality"]["κ"]      = vcat(output_initial_values["seasonality"]["κ"], components["param_$i"]["seasonality"]["hyperparameters"]["κ"])
                    output_initial_values["seasonality"]["γ"]      = cat(output_initial_values["seasonality"]["γ"], components["param_$i"]["seasonality"]["hyperparameters"]["γ"], dims = 3)
                    output_initial_values["seasonality"]["γ_star"] = cat(output_initial_values["seasonality"]["γ_star"], components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"], dims = 3)

                else
                    output_initial_values["seasonality"]["γ"]      = hcat(output_initial_values["seasonality"]["γ"], components["param_$i"]["seasonality"]["hyperparameters"]["γ"])
                    output_initial_values["seasonality"]["γ_star"] = hcat(output_initial_values["seasonality"]["γ_star"], components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"])
                end
            end

            if has_AR(ar, i)
                output_initial_values["ar"]["ϕ"] = hcat(output_initial_values["ar"]["ϕ"], components["param_$i"]["ar"]["hyperparameters"]["ϕ"])
                output_initial_values["ar"]["κ"] = vcat(output_initial_values["ar"]["κ"], components["param_$i"]["ar"]["hyperparameters"]["κ"])
                output_initial_values["ar"]["values"] = hcat(output_initial_values["ar"]["values"],components["param_$i"]["ar"]["value"])
            end
        end
    end

    return convert(Dict{String, Any}, output_initial_values)

end

function initialize_components!(model::Ml, initial_values::Dict{String, Any}, gas_model::GASModel) where {Ml}

    @unpack dist, time_varying_params, d, level, seasonality, ar = gas_model

    set_start_value.(model[:params], round.(initial_values["param"]; digits = 5))
    #set_start_value.(model[:c], round.(initial_values["intercept"]["values"]; digits = 5))
    #println(round.(initial_values["intercept"]["values"]; digits = 5))
    
    if haskey(initial_values, "fixed_param")
        set_start_value.(model[:fixed_params], round.(initial_values["fixed_param"]; digits = 5))
    end
    
    if has_random_walk_slope(level) 
        set_start_value.(model[:RWS], round.(initial_values["rws"]["values"]; digits = 5))
        set_start_value.(model[:κ_RWS], round.(initial_values["rws"]["κ"]; digits = 5))
        set_start_value.(model[:b],  round.(initial_values["slope"]["values"]; digits = 5))
        set_start_value.(model[:κ_b], round.(initial_values["slope"]["κ"]; digits = 5))
    end

    if has_ar1_level(level)
        set_start_value.(model[:AR1_LEVEL], round.(initial_values["ar1_level"]["values"]; digits = 5))
        set_start_value.(model[:κ_AR1_LEVEL], round.(initial_values["ar1_level"]["κ"]; digits = 5))
        set_start_value.(model[:ϕ_AR1_LEVEL],  round.(initial_values["ar1_level"]["ϕ"]; digits = 5))
    end


    if has_random_walk(level) 
        set_start_value.(model[:RW], round.(initial_values["rw"]["values"]; digits = 5))
        set_start_value.(model[:κ_RW], round.(initial_values["rw"]["κ"]; digits = 5))
    end

    if has_seasonality(seasonality)
        seasonality_dict, stochastic = get_seasonality_dict_and_stochastic(seasonality)
        if stochastic
            set_start_value.(model[:κ_S], round.(initial_values["seasonality"]["κ"]; digits = 5))
        end

        if haskey(initial_values["seasonality"], "γ")
            set_start_value.(model[:γ], round.(initial_values["seasonality"]["γ"]; digits = 5))
            set_start_value.(model[:γ_star], round.(initial_values["seasonality"]["γ_star"]; digits = 5))        
        end
    end

    if has_AR(ar)
        set_start_value.(model[:AR], initial_values["ar"]["values"])
        set_start_value.(model[:ϕ], initial_values["ar"]["ϕ"])
        set_start_value.(model[:κ_AR], initial_values["ar"]["κ"])
    end

    if haskey(initial_values, "explanatories")
        set_start_value.(model[:β], initial_values["explanatories"])
    end

end

