
"""
NumDiffTools.jacobian(f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Returns jacobian `J(f)` of `f` evaluted at `x`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function jacobian(f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x)

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    result = Matrix{eltype(fx)}(undef, length(fx), length(x))
    dx = zero(x)
    for i = 1:length(x)
        dx[i] = 1

        result[:, i] .= extrapolate(h->(f(x + h*dx) - fx)/h,          # Function to extrapolate
                                    A1, A2,                           # Work Arrays
                                    hh, k,                            # Sequence of step sizes (h)
                                    abstol, reltol                    # tolerances
                                    )
        dx[i] = 0        
    end
    result
end


"""
NumDiffTools.jacobian(f!, y::AbstractArray, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Returns jacobian `J(f!)` of `f!` evaluted at `x`, assuming `f` is called as `f!(y, x)` where the result is stored in y.
"""
function jacobian(f!, y::AbstractArray, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)

    # Initialize Work Arrays
    A1 = Vector{typeof(y)}(undef, maxiters)
    A2 = Vector{typeof(y)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    fx = similar(y)
    f!(fx, x)
    function A(h, x, dx)
        f!(y, x + h*dx)
        return (y - fx)/h
    end

    result = Matrix{eltype(y)}(undef, length(y), length(x))
    dx = zero(x)
    for i = 1:length(x)
        dx[i] = 1        
        result[:,i] .= extrapolate(h->A(h, x, dx),                   # Function to extrapolate
                                   A1, A2,                           # Work Arrays
                                   hh, k,                            # Sequence of step sizes (h)
                                   abstol, reltol                    # tolerances
                                   )
        dx[i] = 0        
    end
    result
end


"""
NumDiffTools.jacobian!(result::AbstractArray, f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Stores jacobian `J(f)` of `f` evaluted at `x` in `result`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function jacobian!(result::AbstractArray, f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
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

        result[:, i] .= extrapolate(h->(f(x + h*dx) - fx)/h,          # Function to extrapolate
                                    A1, A2,                           # Work Arrays
                                    hh, k,                            # Sequence of step sizes (h)
                                    abstol, reltol                    # tolerances
                                    )
        dx[i] = 0        
    end
    result
end


"""
NumDiffTools.jacobian!(result::AbstractArray, f!, y::AbstractArray, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Stores jacobian `J(f!)` of `f!` evaluted at `x` in `result`, assuming `f` is called as `f!(y, x)` where the result is stored in y.
"""
function jacobian!(result::AbstractArray, f!, y::AbstractArray, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)

    # Initialize Work Arrays
    A1 = Vector{typeof(y)}(undef, maxiters)
    A2 = Vector{typeof(y)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 1:maxiters

    fx = similar(y)
    f!(fx, x)
    function A(h, x, dx)
        f!(y, x + h*dx)
        return (y - fx)/h
    end

    dx = zero(x)
    for i = 1:length(x)
        dx[i] = 1        
        result[:,i] .= extrapolate(h->A(h, x, dx),                   # Function to extrapolate
                                   A1, A2,                           # Work Arrays
                                   hh, k,                            # Sequence of step sizes (h)
                                   abstol, reltol                    # tolerances
                                   )
        dx[i] = 0        
    end
    result
end


