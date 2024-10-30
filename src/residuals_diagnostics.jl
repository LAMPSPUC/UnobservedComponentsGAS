# Function to retrieve residuals from a fitted model
# Parameters:
#   - fitted_model::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to retrieve ("q" for quantile, "std" for standardized, "cs" for conditional score)
# Returns:
#   - The specified residuals as a vector or matrix
function get_residuals(output::Output; type::String="q")
    fitted_model = deepcopy(output)
    if type == "q"
        return fitted_model.residuals["q_residuals"][1:end]
    elseif type == "std"
        return fitted_model.residuals["std_residuals"][1:end]
    else
        return fitted_model.residuals["cs_residuals"][1:end, :]
    end
end

# Function to plot residuals from a fitted model
# Parameters:
#   - fitted_model::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to plot ("q" for quantile, "std" for standardized, "cs" for conditional score)
# Returns:
#   - A plot of the specified residuals
function plot_residuals(fitted_model::Output; type::String="q")

    resid = get_residuals(fitted_model; type=type)

    if type == "q"
        name = "Quantile"
    elseif type == "std"
        name = "Standardized"
    else
        name = "Conditional Score"
    end

    if type == "cs"
        num_params = length(fitted_model.fitted_params)
        plots_cs = []
        for n in 1:num_params
            push!(plots_cs, plot(resid[:,n],  label = "param $n", 
                    title = "param $n", title_loc = :center))
        end
        
        if num_params == 2
            plot(plots_cs[1], plots_cs[2], layout=(2,1), suptitle = "$name Residuals", size=(800,600))
        else
            plot(plots_cs[1], plots_cs[2], plots_cs[3], layout=(3,1), suptitle = "$name Residuals", size=(800,600))
        end

    else
        plot(resid, title = "$name Residuals", label = "")
    end
end

# Function to calculate the autocorrelation function (ACF) of residuals
# Parameters:
#   - fitted_model::Output: The model output object containing residuals
#   - lags::Int=25: The number of lags to include in the ACF calculation
#   - type::String="q": The type of residuals to use ("q" for quantile, "std" for standardized)
#   - squared::Bool=false: Whether to square the residuals before calculating ACF
# Returns:
#   - A vector of ACF values
function get_acf_residuals(fitted_model::Output; lags::Int=25, type::String="q", squared::Bool=false)
    
    resid = get_residuals(fitted_model; type=type)
    println(length(resid))
    squared == true ? resid = resid.^2 : nothing
    
    return StatsBase.autocor(resid, 0:lags)
end

# Function to plot the autocorrelation function (ACF) of residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - lags::Int=25: The number of lags to include in the ACF plot
#   - type::String="q": The type of residuals to plot ("q" for quantile, "std" for standardized)
#   - squared::Bool=false: Whether to square the residuals before calculating ACF
# Returns:
#   - A plot of the ACF of the specified residuals
function plot_acf_residuals(output::Output; lags::Int=25, type::String="q", squared::Bool=false)
    acf_values = get_acf_residuals(output; lags = lags, type = type, squared = squared)
    resid      = get_residuals(output; type=type)
    println(length(resid))
    if type == "q"
        name = "Quantile"
    elseif type == "std"
        name = "Standardized"
    end

    squared ? is_squared = "Squared" : is_squared = ""
    
    lag_values = collect(0:lags)
    conf_interval = 1.96 / sqrt(length(resid))

    plot(title="ACF $name $is_squared Residuals")
    plot!(lag_values, acf_values, seriestype=:stem, label="", xticks=(lag_values, lag_values))
    hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "CI 95%")
end

# Function to plot a histogram of residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to plot ("q" for quantile, "std" for standardized)
#   - bins::Int=20: The number of bins to use in the histogram
# Returns:
#   - A histogram plot of the specified residuals
function plot_histogram(output::Output; type::String="q", bins::Int=20)
    resid = get_residuals(output; type=type)
    
    if type == "q"
        name = "Quantile"
    elseif type == "std"
        name = "Standardized"
    end

    histogram(resid, title="Histogram $name Residuals", label = "", bins = bins)
end

# Function to plot a QQ plot of residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to plot ("q" for quantile, "std" for standardized)
# Returns:
#   - A QQ plot of the specified residuals
function plot_qqplot(output::Output; type::String="q")
    resid = get_residuals(output; type=type)
    
    if type == "q"
        name = "Quantile"
    elseif type == "std"
        name = "Standardized"
    end

    plot(qqplot(Normal, resid), title = "QQPlot $name Residuals")
end

# Function to perform the Jarque-Bera test for normality on residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to test ("q" for quantile, "std" for standardized)
# Returns:
#   - A dictionary with the Jarque-Bera test statistic, p-value, skewness, and kurtosis
function jarquebera(output::Output; type::String="q")
    resid = get_residuals(output; type=type)
    jb = JarqueBeraTest(resid)
    return Dict("stat" => jb.JB, "pvalue" => pvalue(jb), "skew" => jb.skew, "kurt" => jb.kurt)
end

# Function to perform the Ljung-Box test for autocorrelation on residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to test ("q" for quantile, "std" for standardized)
#   - squared::Bool=false: Whether to square the residuals before testing
#   - lags::Int=25: The number of lags to include in the test
# Returns:
#   - A dictionary with the Ljung-Box test statistic and p-value
function ljungbox(output::Output; type::String="q", squared::Bool=false, lags::Int = 25)
    resid = get_residuals(output; type=type)

    squared ? resid = resid.^2 : nothing

    lb = LjungBoxTest(resid, lags)
    return Dict("stat" => lb.Q, "pvalue" => pvalue(lb))
end

# Function to perform the H test for variance on residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to test ("q" for quantile, "std" for standardized)
# Returns:
#   - A dictionary with the F-statistic and p-value of the H test
function Htest(output::Output; type::String="q")
    resid = get_residuals(output; type=type)

    T = length(resid)
    h = Int64(floor(T/3))

    res1 = resid[1:h]
    res3 = resid[2*h+1:end]

    σ2_1 = var(res1; corrected=true)
    σ2_3 = var(res3; corrected=true)

    df1 = length(res1) - 1
    df3 = length(res3) - 1

    F_statistic = (σ2_3 / σ2_1)

    p_value = 2* minimum([ccdf(FDist(df3, df1), F_statistic), cdf(FDist(df3, df1), F_statistic)])

    return Dict("stat" => F_statistic, "pvalue" => p_value)
end

# Function to perform the ARCH test for autoregressive conditional heteroskedasticity on residuals
# Parameters:
#   - output::Output: The model output object containing residuals
#   - type::String="q": The type of residuals to test ("q" for quantile, "std" for standardized)
#   - lags::Int=25: The number of lags to include in the test
# Returns:
#   - A dictionary with the ARCH test statistic and p-value
function archtest(output::Output; type::String="q", lags::Int=25)
    resid = get_residuals(output; type=type)
    arch = ARCHLMTest(resid, lags)
    return Dict("stat" => arch.LM, "pvalue" => pvalue(arch))
end

# Function to get p-values of multiple residuals diagnostics tests
# Parameters:
#   - output::Output: The model output object containing residuals
#   - lags::Int=25: The number of lags to include in the tests
#   - type::String="q": The type of residuals to test ("q" for quantile, "std" for standardized)
# Returns:
#   - A dictionary with p-values from various residuals diagnostics tests
function get_residuals_diagnosis_pvalues(output::Output; lags::Int=25, type::String="q")
    jb   = jarquebera(output; type = type)
    lb   = ljungbox(output; type = type, squared = false, lags = lags)
    lb2  = ljungbox(output; type = type, squared = true, lags = lags)
    H    = Htest(output; type = type)
    arch = archtest(output; type = type, lags = lags)
    
    return Dict("JarqueBera" => jb["pvalue"], "HVariance" => H["pvalue"],
                "LjungBox" => lb["pvalue"], "LjungBoxSquared" => lb2["pvalue"],
                "ARCH" => arch["pvalue"])
end
