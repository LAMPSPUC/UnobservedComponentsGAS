"
Creates the decision variables for the distribution parameters. 
For the time varying parameters, creates a vector of decision variables with length = T.
For the fixed parameters, creates a unique decision variable.
Also adds the positivity constraints, when necessary.
Returns a matrix of dimentions T x num. parameters associated to the models decision variables.
"
function include_parameters(model::Ml, time_varying_params::Vector{Bool}, T::Int64, dist::ScoreDrivenDistribution, fixed_ν::Union{Missing, Int64}) where {Ml}

    num_params              = get_num_params(dist)
    positive_constrants     = check_positive_constrainst(dist)
    idx_time_varying_params = get_idxs_time_varying_params(time_varying_params)
    idx_fixed_params        = setdiff(1:num_params, idx_time_varying_params)
    #num_time_varying_params = length(time_varying_params) 

    @variable(model, params[1:T, idx_time_varying_params])
    @variable(model, fixed_params[idx_fixed_params])

    for i in 1:num_params
        #Verifica se um parametro variante no tempo precisa dessa restrição
        if positive_constrants[i] * time_varying_params[i] == 1
            @constraint(model, [t = 1:T], params[t, i] ≥ 1e-4)
        end
        #Verifica se um parametro fixo precisa dessa restrição 
        if positive_constrants[i] * !time_varying_params[i] == 1
            @constraint(model, fixed_params[i] ≥ 1e-4)
        end
    end

    if typeof(dist) == tLocationScaleDistribution
        JuMP.fix(model[:fixed_params][end], fixed_ν)
    end

    parameters = Matrix(undef, T, num_params)

    for i in 1:num_params
        
        if i in idx_time_varying_params
            parameters[:, i] = model[:params][:, i]
        else
            parameters[:, i] .= model[:fixed_params][i]
        end
    end

    return parameters
end

"
Creates the expression for the score used in the parameters dynamic.
"
function compute_score(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, d::Float64, time_varying_params::Vector{Bool}, T::Int64, dist::ScoreDrivenDistribution) where {Ml, Gl, Fl}

    idx_time_varying_params = get_idxs_time_varying_params(time_varying_params)
    num_time_varying_params = length(idx_time_varying_params)

    num_param = get_num_params(dist)
    dist_code = get_dist_code(dist)

    s = Vector(undef, num_param)

    if num_param == 2
        register(model, :scaled_score, 6, scaled_score; autodiff = true)
        for i in idx_time_varying_params
            s[i] = @NLexpression(model,[t = 2:T], scaled_score(parameters[t-1, 1], parameters[t-1, 2], y[t-1], d, dist_code, i))
        end
    elseif num_param == 1
        #IMPLEMENTAR SCALED SCORE FUNCTION PARA 1 PARAMETRO
    else
        register(model, :scaled_score, 7, scaled_score; autodiff = true)
        for i in idx_time_varying_params
            s[i] = @NLexpression(model,[t = 2:T], scaled_score(parameters[t-1, 1], parameters[t-1, 2], parameters[t-1, 3], y[t-1], d, dist_code, i))
        end
    end
  
    return s
end

"
Include the decision variables associated with the exogenous variables.
"
function include_explanatory_variables!(model::Ml, X::Matrix{Fl}) where {Ml, Fl}

    T, p = size(X)
    @variable(model, β[1:p])
end

"
Includes the complete dynamic for the time-varying parameters in the JuMP model.
"
function include_dynamics!(model::Ml, parameters::Matrix{Gl}, gas_model::GASModel, X::Union{Matrix, Missing}, T::Int64) where {Ml, Gl}

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model
    
    idx_time_varying_params = get_idxs_time_varying_params(time_varying_params) 

    @variable(model, c[idx_time_varying_params])

    has_explanatory = !ismissing(X) ? true : false

    for i in idx_time_varying_params

        dynamic_aux = Vector(undef, T)

        has_explanatory_param = has_explanatory && i == 1

        for t in 2:T
            dynamic_aux[t] = model[:c][i] + 
                             include_component_in_dynamic(model, :RW, has_random_walk(random_walk, i), t, i) +
                             include_component_in_dynamic(model, :RWS, has_random_walk_slope(random_walk_slope, i), t, i) +
                             include_component_in_dynamic(model, :AR, has_AR(ar, i), t, i) +
                             include_component_in_dynamic(model, :S, has_seasonality(seasonality, i), t, i) + 
                             include_explanatories_in_dynamic(model, X, has_explanatory_param, t, i)

        end
        @NLconstraint(model,[t = 2:T], parameters[t, i] ==  dynamic_aux[t])
    end

end

"
Includes the objective function in the JuMP model.
"
function include_objective_function!(model::Ml, parameters::Matrix{Gl}, y::Vector{Fl}, T::Int64, robust::Union{Float64, Bool}, dist_code::Int64; α::Float64 = 0.5, robust_prop::Float64 = 0.7) where {Ml, Gl, Fl}
    
    dist_name = DICT_CODE[dist_code]
    num_params = size(parameters, 2)
    κ_variables = filter(v -> occursin(r"^κ", string(v)), all_variables(model))
    
    if robust
        #@info("Defining Robust Model's Object Function")
        k = Int64(floor(robust_prop * T))
        @variable(model, δ ≥ 1e-4)
        @variable(model, u[1:T] ≥ 1e-4)
        
        if num_params == 3
            #register(model, :log_pdf, 4, DICT_LOGPDF[dist_name]; autodiff = true)
            @NLconstraint(model, [t = 2:T], δ + u[t] ≥  - (log_pdf(parameters[t, 1], parameters[t, 2], parameters[t, 3], y[t])))
        elseif num_params == 2
            #register(model, :log_pdf, 3, DICT_LOGPDF[dist_name]; autodiff = true)
            @NLconstraint(model, [t = 2:T], δ + u[t] ≥  - (log_pdf(parameters[t, 1], parameters[t, 2], y[t])))
        end
        @NLobjective(model, Min, (1 - α) * (δ*k + sum(u[t] for t in 2:T)) + α * sum(κ_variables[i]^2 for i in eachindex(κ_variables)))

    else
        #@info("Defining Model's Object Function")
        if num_params == 3
            #register(model, :log_pdf, 4, DICT_LOGPDF[dist_name]; autodiff = true)
            @NLobjective(model, Min, (1 - α) * (-sum(log_pdf(parameters[t, 1], parameters[t, 2], parameters[t, 3], y[t]) for t in 2:T)) + α * sum(κ_variables[i]^2 for i in eachindex(κ_variables)))
        elseif num_params == 2
            #register(model, :log_pdf, 3, DICT_LOGPDF[dist_name]; autodiff = true)
            @NLobjective(model, Min, (1 - α) * (-sum(log_pdf(parameters[t, 1], parameters[t, 2], y[t]) for t in 2:T)) + α * sum(κ_variables[i]^2 for i in eachindex(κ_variables)))
        end    
    end
    
end

