
"""
scaled_score(first_param, second_param, y, d, dist_code, which_param)

Calculate the scaled score taking into account the specified parameters, observation, distribution (with two parameters) and scale parameter d.

    # Arguments
    - `first_param`: Value of the first parameter of the chosen distribution.
    - `second_param`: Value of the second parameter of the chosen distribution.
    - `y`: Observed data.
    - `d`: Scale parameter (d ∈ {0.0, 0.5, 1.0}).
    - `dist_code`: The code that specifies the chosen distribution.
    - `which_param`: Index of the score vector associated with the parameter of interest.

    # Returns
    - The element of interest in the score vector considering the specified arguments.
"""

function scaled_score(first_param, second_param, y, d, dist_code, which_param)

    dist_name = DICT_CODE[dist_code]

    ∇ = DICT_SCORE[dist_name](first_param, second_param, y)
    
    if d == 0.0 
        s = Matrix(I, length(∇), length(∇))' * ∇

    elseif d == 0.5
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param)
        try
            s = cholesky(inv(FI), check = false).UL' * ∇
        catch
            s = cholesky(pinv(FI), check = false).UL' * ∇
        end


    else 
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param)
        try
            s = inv(FI) * ∇
        catch
            s = pinv(FI) * ∇
        end

    end

    if which_param == 1
        return s[1]
    else
        return s[2]
    end
end

"""
scaled_score(first_param, second_param,third_param, y, d, dist_code, which_param)

Calculate the scaled score taking into account the specified parameters, observation, distribution (with two parameters) and scale parameter d.

    # Arguments
    - `first_param`: Value of the first parameter of the chosen distribution.
    - `second_param`: Value of the second parameter of the chosen distribution.
    - `third_param`: Value of the third parameter of the chosen distribution.
    - `y`: Observed data.
    - `d`: Scale parameter (d ∈ {0.0, 0.5, 1.0}).
    - `dist_code`: The code that specifies the chosen distribution.
    - `which_param`: Index of the score vector associated with the parameter of interest.

    # Returns
    - The element of interest in the score vector considering the specified arguments.
"""
function scaled_score(first_param, second_param, third_param, y, d, dist_code, which_param)

    dist_name = DICT_CODE[dist_code]

    ∇ = DICT_SCORE[dist_name](first_param, second_param, third_param, y)
    
    if d == 0.0 
        s = Matrix(I, length(∇), length(∇))' * ∇

    elseif d == 0.5
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param, third_param)
        try
            s = cholesky(inv(FI), check = false).UL' * ∇
        catch
            s = cholesky(pinv(FI), check = false).UL' * ∇
        end


    else
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param, third_param)
        try
            s = inv(FI) * ∇
        catch
            s = pinv(FI) * ∇
        end
    end

    if which_param == 1
        return s[1]
    elseif which_param == 2
        return s[2]
    else 
        return s[3]
    end
end