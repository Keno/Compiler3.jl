using Compiler3
using Compiler3: StaticSubGraph

# A semi-interesting, fully static function
exp_kernel(x::Float64) = @Base.Math.horner(x, 1.66666666666666019037e-1,
    -2.77777777770155933842e-3, 6.61375632143793436117e-5,
    -1.65339022054652515390e-6, 4.13813679705723846039e-8)

ei, ssg = Compiler3.analyze(Tuple{typeof(exp_kernel), Float64})
