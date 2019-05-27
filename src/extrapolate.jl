#######################################
# Implements Richardson Extrapolation #
#######################################

import LinearAlgebra: norm


"""
Implements a single iteration of Richardson extrapolation

i (input) = Row of Richardson Extrapolation
Ai1 (output) = Next row of Richardson Extrapolation to be calculated
Ai (input) = Previous Row of Richardson Extrapolation
k (inputs) = vector of error orders
t = ratio of step sizes between the i and i-1 iteration
"""
function extrapolate_iter(A::Function, i::Integer, Ai::AbstractArray, Ai1::AbstractArray, h::AbstractArray, k::AbstractArray, abstol::Real, reltol::Real, erri::Real)
    Ai1[1] = A(h[i+1])
    t = h[i]/h[i+1]
    for j = 1:i
        tk = t^k[j]
        Ai1[j+1] = (tk*Ai1[j] - Ai[j])/(tk- 1)
    end    
    # We estimate the error by comparing the estimated value at the previous and current row
    # This gives a good estimate of the error at the previous iteration,
    # so we extrapolate to estimate the error at the current order
    err = norm(Ai[i] - Ai1[i+1]) * (h[i+1]^k[i+1]) / (h[i]^k[i])

    # Checking for stopping conditions
    if err < abstol + reltol*norm(Ai1[i+1]) || length(Ai1) == i+1
        # If the err is acceptable or we have reached the maximum number of iterations, return the answer and the error
        return Ai1[i+1], err
    elseif err > erri
        # If the estimate error has increased, we are probably running into round-off error and should stop
        # We return the previous answer and error estimate
        return Ai[i], erri
    else
        # Otherwise, we calculate another iteration
        # We swap the order of the work arrays Ai, Ai1
        return extrapolate_iter(A, i+1, Ai1, Ai, h, k, abstol, reltol, err)
    end
end

function extrapolate(A::Function, A1::AbstractArray, A2::AbstractArray, h::AbstractArray, k::AbstractArray, abstol::Real, reltol::Real)
    A1[1] = A(h[1])
    ans, err = extrapolate_iter(A, 1, A1, A2, h, k, abstol, reltol, Inf)
    if err > abstol + reltol*norm(ans)
        @warn @sprintf("Error estimate of derivative larger than requested tolerance.  Err: %g, abstol + reltol*|ans|: %g", err, abstol + reltol*norm(ans))
    end
    return ans
end
