"""
# update_fitted_params_and_components_dict(gas_model::GASModel, output::Output, new_y::Vector{Fl}, new_X::Union{Missing, Matrix{Float64}}; initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl}

Updates the fitted parameters and components dictionary based on new data.

## Arguments
- `gas_model::GASModel`: The GAS model containing parameters and specifications.
- `output::Output`: The output structure containing the fitted values, residuals, and information criteria.
- `new_y::Vector{Fl}`: The vector of new observed values for the time series data.
- `new_X::Union{Missing, Matrix{Float64}}`: The matrix of new explanatory variables. Use `missing` if no new explanatory variables are available.
- `initial_values::Union{Dict{String, Any}, Missing}`: Optional. A dictionary containing initial values for updating parameters. Default is `missing`.

## Returns
- `new_output::Output`: The updated output structure containing the modified fitted parameters and components.
"""
function update_fitted_params_and_components_dict(gas_model::GASModel, output::Output, new_y::Vector{Fl}, new_X::Union{Missing, Matrix{Float64}};
                                                    initial_values::Union{Dict{String, Any}, Missing} = missing) where {Fl}

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    components    = output.components
    fitted_params = output.fitted_params
    dist_code     = get_dist_code(dist)
    num_params    = get_num_params(dist)

    idx_params                    = get_idxs_time_varying_params(time_varying_params)
    num_harmonic, seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)

    T = length(new_y)

    new_output = deepcopy(output)

    new_components    = deepcopy(components)
    new_fitted_params = deepcopy(fitted_params)

    for i in 1:num_params

        if time_varying_params[i]
            new_fitted_params["param_$i"]    = zeros(T)

            if ismissing(initial_values)
                new_fitted_params["param_$i"][1] = fitted_params["param_$i"][1]
            else
                new_fitted_params["param_$i"][1] = initial_values["param"][1]  ## CHECAR
            end

            if has_random_walk(random_walk, i)
                new_components["param_$i"]["level"]["value"]    = zeros(T)

                if ismissing(initial_values)
                    new_components["param_$i"]["level"]["value"][1] = components["param_$i"]["level"]["value"][1]
                else
                    new_components["param_$i"]["level"]["value"][1] = initial_values["rw"]["values"][1]
                end
            end

            if has_random_walk_slope(random_walk_slope, i)
                new_components["param_$i"]["slope"]["value"] = zeros(T)
                new_components["param_$i"]["level"]["value"] = zeros(T)

                if ismissing(initial_values)
                    new_components["param_$i"]["slope"]["value"][1] = components["param_$i"]["slope"]["value"][1]
                    new_components["param_$i"]["level"]["value"][1] = components["param_$i"]["level"]["value"][1]
                else
                    new_components["param_$i"]["slope"]["value"][1] = initial_values["slope"]["values"][1]
                    new_components["param_$i"]["level"]["value"][1] = initial_values["rws"]["values"][1]
                end
            end

            if has_seasonality(seasonality, i)
                new_components["param_$i"]["seasonality"]["value"] = zeros(T)

                if ismissing(initial_values)
                    new_components["param_$i"]["seasonality"]["value"][1] = components["param_$i"]["seasonality"]["value"][1]
                else
                    new_components["param_$i"]["seasonality"]["value"][1] = initial_values["seasonality"]["values"][1]
                end
            end
        else
            new_fitted_params["param_$i"] = ones(T) .* fitted_params["param_$i"][1]
        end
    end

    if !ismissing(new_X)
        explanatories_dynamic = new_X * components["param_1"]["explanatories"]
    else
        explanatories_dynamic = zeros(T)
    end

    score = zeros(T, num_params)

    # Updating the dynamic for each time t and time-varying parameter
    for t in 2:T
        @info("Updating results at t = $t")
        for i in idx_params
            
            if num_params == 2
                score[t, i] = scaled_score(new_fitted_params["param_1"][t - 1], 
                                            new_fitted_params["param_2"][t - 1], 
                                            new_y[t - 1], d, dist_code, i)
            end

            if num_params == 3
                score[t, i] = scaled_score(new_fitted_params["param_1"][t - 1], 
                                            new_fitted_params["param_2"][t - 1],
                                            new_fitted_params["param_3"][t - 1], 
                                            new_y[t - 1], d, dist_code, i)
            end
            
            new_fitted_params["param_$i"][t] = components["param_$i"]["intercept"]

            if has_random_walk(random_walk, i)
                κ_rw  = components["param_$i"]["level"]["hyperparameters"]["κ"]

                new_components["param_$i"]["level"]["value"][t] = new_components["param_$i"]["level"]["value"][t - 1] + κ_rw * score[t, i]

                new_fitted_params["param_$i"][t] += new_components["param_$i"]["level"]["value"][t]
            end

            if has_random_walk_slope(random_walk_slope, i)
                κ_rws = components["param_$i"]["level"]["hyperparameters"]["κ"]
                κ_b   = components["param_$i"]["slope"]["hyperparameters"]["κ"]

                new_components["param_$i"]["slope"]["value"][t] = new_components["param_$i"]["slope"]["value"][t - 1] + κ_b * score[t, i]
                new_components["param_$i"]["level"]["value"][t] = new_components["param_$i"]["level"]["value"][t - 1] +
                                                                    new_components["param_$i"]["slope"]["value"][t - 1] + κ_rws * score[t, i]

                new_fitted_params["param_$i"][t] += new_components["param_$i"]["level"]["value"][t]
            end

            if has_seasonality(seasonality, i)
                unique_num_harmonic = unique(num_harmonic)[1]

                # κ_S    = components["param_$i"]["seasonality"]["hyperparameters"]["κ"]
                γ      = components["param_$i"]["seasonality"]["hyperparameters"]["γ"]
                γ_star = components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"]

                new_components["param_$i"]["seasonality"]["value"][t] = sum(γ[j, i]*cos(2 * π * j * t/seasonal_period[i]) + 
                                                                        γ_star[j, i] * sin(2 * π * j * t/seasonal_period[i]) for j in 1:unique_num_harmonic)# + κ_S * score[t, i]
            
                new_fitted_params["param_$i"][t] += new_components["param_$i"]["seasonality"]["value"][t]
            end

            if i == 1
                new_fitted_params["param_$i"][t] += explanatories_dynamic[t]
            end
        end
    end

    new_output.components    = new_components
    new_output.fitted_params = new_fitted_params

    return new_output
end
