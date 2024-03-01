
"
Defines the scaled score for a distribution with two parameters.
"
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


"
Defines the scaled score for a distribution with three parameters.
"
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