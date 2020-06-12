using ChainRules
using StaticArrays

include("zygoteng.jl")

# Scalar AD example case
function rober1(u, p)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    -k₁*y₁+k₃*y₂*y₃
end

struct Chain{T <: Tuple}
    layers::T
end
Chain(layers...) = Chain(layers)
apply_chain(layer::Tuple{}, x) = x
apply_chain(layers, x) = apply_chain(Base.tail(layers), getfield(layers, 1)(x))
(c::Chain)(x) = apply_chain(c.layers, x)

struct Dense{F,S,T}
    W::S
    b::T
    σ::F
end

function (a::Dense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    σ.(W*x .+ b)
end


g(NNODE,t,x,y) = t*(1-x)*x*(1-y)*y*NNODE(@SVector [t,x,y]) + sin(2π*y)*sin(2π*x)
const dx = 0.1f0
function loss(NNODE)
    gridvalues = dx:dx:(1-dx)
    l = 0.0
    for t in gridvalues, x in gridvalues, y in gridvalues
        l += (x->g(NNODE, t, x, y))''(x) +
             (y->g(NNODE, t, x, y))''(y) -
             (t->g(NNODE, t, x, y))'(t) +
             sin(2π*t)*sin(2π*x)*sin(2π*y)
    end
    l
end

# Good enough for testing
Dense(a::Integer, b::Integer, σ) =
    Dense(rand(Float32, b, a), rand(Float32, b), σ)

NNODE = Chain(Dense(3,256,tanh),
              Dense(256,256,tanh),
              Dense(256,256,tanh),
              Dense(256,1, identity),x->x[1])

# Don't fall over on this semi-complicated nested AD case
training_step(NNODE) = gradient(NNODE->loss(NNODE), NNODE)
