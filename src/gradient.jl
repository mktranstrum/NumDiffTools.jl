
"""
NumDiffTools.gradient(f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Returns gradient `∇f` of `f` evaluted at `x`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function gradient(f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x)

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    result = Vector{typeof(fx)}(undef, length(x))
    dx = zero(x)
    for i = 1:length(x)
        dx[i] = 1

        result[i] = extrapolate(h->(f(x + h*dx) - fx)/h,          # Function to extrapolate
                                A1, A2,                           # Work Arrays
                                hh, k,                            # Sequence of step sizes (h)
                                abstol, reltol                    # tolerances
                                )
        dx[i] = 0        
    end
    result
end

"""
NumDiffTools.gradient!(result::AbstractArray, f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Stores gradient `∇f` of `f` evaluted at `x` in `result`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function gradient!(result::AbstractArray, f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x)

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    dx = zero(x)
    for i = 1:length(x)
        dx[i] = 1

        result[i] = extrapolate(h->(f(x + h*dx) - fx)/h,          # Function to extrapolate
                                A1, A2,                           # Work Arrays
                                hh, k,                            # Sequence of step sizes (h)
                                abstol, reltol                    # tolerances
                                )
        dx[i] = 0        
    end
    nothing 
end
