
"""
NumDiffTools.derivative(f, x::Real, [abstol, reltol, h, t, maxiters])

Returns `df/dx` evaluted at `x`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function derivative(f, x::Real; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x);

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    extrapolate(h->(f(x + h) - fx)/h,             # Function to extrapolate
                A1, A2,                           # Work Arrays
                hh, k,                            # Sequence of step sizes (h)
                abstol, reltol                    # tolerances
                )
end


"""
NumDiffTools.derivative(f!, y::AbstractArray, x::Real)

Returns `df!/dx` evaluated at `x` assuming `f!` is called as `f!(y,x)`.  Result is also stored in `y`.

"""
function derivative(f!, y::AbstractArray, x::Real; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    # Initialize Work Arrays
    A1 = Vector{typeof(y)}(undef, maxiters)
    A2 = Vector{typeof(y)}(undef, maxiters)
    
    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    fx = similar(y)
    f!(fx,x)
    # Function to extrapolate
    function A(h, x)
        f!(y, x + h)
        return (y - fx)/h
    end
    y .= extrapolate(h->A(h,x),                        # Function to extrapolate
                     A1, A2,                           # Work Arrays
                     hh, k,                            # Sequence of step sizes (h)
                     abstol, reltol                    # tolerances
                     )    
end

"""
NumDiffTools.derivative(result::AbstractArray, f, x::Real, [abstol, reltol, h, t, maxiters])

Stores `df/dx` evaluted at `x` in `result`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function derivative!(result::AbstractArray, f, x::Real; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    result .= derivative(f, x; abstol = abstol, reltol = reltol, h = h, t= t, maxiters = maxiters)
    nothing
end

"""
NumDiffTools.derivative!(result::AbstractArray, f!, y::AbstractArray, x::Real)

Stores `df!/dx` evaluated at `x`  in `result` assuming `f!` is called as `f!(y,x)`.  Result is also stored in `y`.
"""
function derivative!(result::AbstractArray, f!, y::AbstractArray, x::Real; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    result .= derivative(f!, y, x; abstol = abstol, reltol = reltol, h = h, t= t, maxiters = maxiters)
    nothing
end

