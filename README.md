# NumDiffTools.jl

NumDiffTools implements methods to estimate derivatives of callable objects using **finite differences** and **Richardson extrapolation**.

The package includes methods for estimating **derivatives**, **gradients**, **Jacobians**, and **Hessians**.

The API mimics that of the ForwardDiff.jl package but the algorithms are heavily influenced by the python package of the same name.  The anticipated use case for this package is to provide similar functionality to ForwardDiff.jl for cases in which automatic differentiation cannot be reasonably applied or may lead to inaccurate or slow results.

Here is a simple example showing the package in action

```julia
julia> using NumDiffTools

julia> f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

julia> x = rand(5) # small size for example's sake
5-element Array{Float64,1}:
 0.978485396861819  
 0.7588076734125035 
 0.2121526391186519 
 0.09619325766415088
 0.5737085901416124 


julia> g = x -> NumDiffTools.gradient(f, x; maxiters = 8); # g = ∇f

julia> g(x)
5-element Array{Float64,1}:
 0.7062943097317391
 0.8649539956525417
 1.3096529516890876
 1.696657740918219 
 0.9930601101808866

julia> NumDiffTools.hessian(f, x; h = 0.05)
5×5 Array{Float64,2}:
 -0.381742   0.319901  0.763416  1.61412    0.35167 
  0.319901  -0.408143  0.717858  1.51817    0.330595
  0.763416   0.717858  0.074826  3.62175    0.789156
  1.61412    1.51817   3.62175   0.513081   1.66907 
  0.35167    0.330595  0.789156  1.66907   -0.317018
```

# TODO

1. Add functionality to preallocate work arrays.
