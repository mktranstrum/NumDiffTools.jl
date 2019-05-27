
import NumDiffTools
norm(x) = maximum(abs.(x))
using Test
using ForwardDiff

@info "Checking derivative(f,x) of scalar function"
err = NumDiffTools.derivative(sin, 2.0) - cos(2.0)
@test norm(err) < 1e-12

@info "Checking derivative(f,x) of vector function"
f(x) = [sin(x), cos(x)]
f!(y,x) = begin
    y[1] = sin(x)
    y[2] = cos(x)
    nothing
end
df(x) = [cos(x), -sin(x)]
err = NumDiffTools.derivative(f, 2.0) - df(2.0)
@test norm(err) < 1e-12


@info "Checking derivative(f!, y, x)"
y = zeros(2)
err =  NumDiffTools.derivative(f!, y, 2.0) - df(2.0)
@test norm(err) < 1e-12


@info "Checking derivative!(result, f, x)"
result = zeros(2)
NumDiffTools.derivative!(result, f, 2.0)
err = result - df(2)
@test norm(err) < 1e-12

@info "Checking derivative!(result, f, y, x)"
NumDiffTools.derivative!(result,f!, y, 2.0)
err = result - df(2.0)
@test norm(err) < 1e-12

@info "Checking gradient(f,x)"
f(x) = prod(cos.(x).^2 - sin.(x).^2)
gradf(x) = [ - 4*cos(x[1])*sin(x[1])*(cos(x[2])^2 - sin(x[2])^2),
             - 4*cos(x[2])*sin(x[2])*(cos(x[1])^2 - sin(x[1])^2)
             ]
err = NumDiffTools.gradient(f, [1.5, 2.5]) - gradf([1.5, 2.5])
@test norm(err) < 1e-12

@info "Checking gradient!(result, f, x)"
NumDiffTools.gradient!(result, f, [1.5, 2.5])
err = result - gradf([1.5, 2.5])
@test norm(err) < 1e-12

@info "checking jacobian(f, x)"
f(x) = [ prod(cos.(x).^2 - sin.(x).^2),
         sum(cos.(x).^2 - sin.(x).^2)]
f!(y, x) = begin
    y .= f(x)
    nothing
end
J(x) = [-4*cos(x[1])*sin(x[1])*(cos(x[2])^2 - sin(x[2])^2)  -4*cos(x[2])*sin(x[2])*(cos(x[1])^2 - sin(x[1])^2);
        (-2*cos(x[1])*sin(x[1]) - 2*sin(x[1])*cos(x[1]))  (-2*cos(x[2])*sin(x[2]) - 2*sin(x[2])*cos(x[2]))
        ]
err = NumDiffTools.jacobian(f, [1.5, 2.5]) - J([1.5, 2.5])
@test norm(err) < 1e-12

@info "checking jacobian(f!, y, x)"
err = NumDiffTools.jacobian(f!, y, [1.5, 2.5]) - J([1.5, 2.5])
@test norm(err) < 1e-12

@info "checking jacobian!(result, f, x)"
result = zeros(2,2)
NumDiffTools.jacobian!(result, f, [1.5, 2.5])
err = result - J([1.5, 2.5])
@test norm(err) < 1e-12

@info "Checking jacobian!(result, f!, y, x)"
result = zeros(2,2)
NumDiffTools.jacobian!(result, f!, y, [1.5, 2.5])
err = result - J([1.5, 2.5])
@test norm(err) < 1e-12

@info "Checking hessian(f, x) for scalar functions of real numbers"
err = NumDiffTools.hessian(sin, 1.0) + sin(1.0)
@test norm(err) < 1e-10

@info "checkking hessian(f, x) for scalar functions of vectors"
f(x) = prod(cos.(x).^2 - sin.(x).^2)

err = NumDiffTools.hessian(f, [1.5, 2.5]) - ForwardDiff.hessian(f, [1.5, 2.5])
@test norm(err) < 1e-10

@info "Checking hessian!(result, f, x)"
result = zeros(2,2)
NumDiffTools.hessian!(result, f, [1.5, 2.5])
err = result - ForwardDiff.hessian(f, [1.5, 2.5])
@test norm(err) < 1e-10

