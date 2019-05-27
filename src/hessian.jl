import LinearAlgebra: Symmetric

"""
NumDiffTools.hessian(f, x::Real, [abstol, reltol, h, t, maxiters])

Returns `H(f)`, i.e., the Hessian, evaluted at `x`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function hessian(f, x::Real; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x);

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 2:2:2*maxiters

    extrapolate(h->(f(x + h) + f(x - h) - 2*fx)/h^2,             # Function to extrapolate
                A1, A2,                                          # Work Arrays
                hh, k,                                           # Sequence of step sizes (h)
                abstol, reltol                                   # tolerances
                )
end

"""
NumDiffTools.hessian(f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Returns `H(f)`, i.e., the Hessian, evaluted at `x`, assuming `f` is called as `f(x)`.
"""
function hessian(f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x);

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 2:2:2*maxiters
    result = Matrix{typeof(fx)}(undef, length(x), length(x)) 
    dx1 = zero(x)
    dx2 = zero(x)
    for i = 1:length(x)
        # Diagonal element
        dx1[i] = 1
        result[i,i] =  extrapolate(h->(f(x + h*dx1) + f(x - h*dx1) - 2*fx)/h^2,             # Function to extrapolate
                A1, A2,                                                                     # Work Arrays
                hh, k,                                                                      # Sequence of step sizes (h)
                abstol, reltol                                                              # tolerances
                )
        for j = (i+1):length(x)
            dx2[j] = 1
            result[i,j] =  extrapolate(h->(f(x + h*dx1 + h*dx2) - f(x + h*dx1 - h*dx2) - f(x - h*dx1 + h*dx2) + f(x - h*dx1 - h*dx2))/(4*h^2),
                                       A1, A2,                                                                     # Work Arrays
                                       hh, k,                                                                      # Sequence of step sizes (h)
                                       abstol, reltol                                                              # tolerances
                                       )
            result[j,i] = result[i,j]
            dx2[j] = 0
        end
        dx1[i] = 0
    end    
    result
end


"""
NumDiffTools.hessian!(result::AbstractArray, f, x::AbstractArray, [abstol, reltol, h, t, maxiters])

Stores `H(f)`, i.e., the Hessian, evaluted at `x` in `result`, assuming `f` is called as `f(x)`.

Optional Arguments:\n
`abstol`: Target absolute error in derivative estimate (default `1e-12`)\n
`reltol`: Target relative error in derivative estimate (default `1e-12`)\n
`h`: Initial step size (default `0.1`)\n
`t`: Ratio of step sizes for each Richardson iteration (default `2.0001`.  Should not be an exact integer for periodic functions)\n
`maxiters`: Maximum number of Richardson iterations to perform (default `6`)\n
"""
function hessian!(result::AbstractArray, f, x::AbstractArray; abstol::Real = 1e-12, reltol::Real = 1e-12, h::Real = 0.1, t::Real = 2.0001, maxiters::Integer = 6)
    fx = f(x);

    # Initialize Work Arrays
    A1 = Vector{typeof(fx)}(undef, maxiters)
    A2 = Vector{typeof(fx)}(undef, maxiters)

    # Vector of step sizes
    hh = [h/t^(i - 1) for i = 1:maxiters]
    # Vector of orders
    k = 2:2:2*maxiters

    dx1 = zero(x)
    dx2 = zero(x)
    for i = 1:length(x)
        # Diagonal element
        dx1[i] = 1
        result[i,i] =  extrapolate(h->(f(x + h*dx1) + f(x - h*dx1) - 2*fx)/h^2,             # Function to extrapolate
                A1, A2,                                                                     # Work Arrays
                hh, k,                                                                      # Sequence of step sizes (h)
                abstol, reltol                                                              # tolerances
                )
        for j = (i+1):length(x)
            dx2[j] = 1
            result[i,j] =  extrapolate(h->(f(x + h*dx1 + h*dx2) - f(x + h*dx1 - h*dx2) - f(x - h*dx1 + h*dx2) + f(x - h*dx1 - h*dx2))/(4*h^2),
                                       A1, A2,                                                                     # Work Arrays
                                       hh, k,                                                                      # Sequence of step sizes (h)
                                       abstol, reltol                                                              # tolerances
                                       )
            result[j,i] = result[i,j]
            dx2[j] = 0
        end
        dx1[i] = 0
    end    
end
