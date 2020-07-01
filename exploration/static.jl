using Compiler3
using Compiler3: StaticSubGraph
using Core.Compiler: MethodInstance
using GPUCompiler

# A semi-interesting, fully static function
exp_kernel(x::Float64) = @Base.Math.horner(x, 1.66666666666666019037e-1,
    -2.77777777770155933842e-3, 6.61375632143793436117e-5,
    -1.65339022054652515390e-6, 4.13813679705723846039e-8)

# Use new compiler infrastructure to infer with separate cache. Likely,
# we may want to set some more aggressive parameters here. We also collect
# inference remarks to emit error messages
Compiler3.analyze_static(Tuple{typeof(exp_kernel), Float64})

# If everything went ok there, do inference again, but this time with
# optimizations. Ideally, we'd be able to just run the optimizer on the
# output of the previous call, but that's not easy right now
ei, ssg = Compiler3.analyze(Tuple{typeof(exp_kernel), Float64}; optimize=true)

# A callback for the compiler to look up results in our separate cache
function cache_lookup(mi::MethodInstance, min_world, max_world)
    return ei.code[mi]
end

params = Base.CodegenParams(;lookup = @cfunction(cache_lookup, Any, (Any, UInt, UInt)))

# generate .o
native_code = ccall(:jl_create_native, Ptr{Cvoid},
    (Vector{Core.MethodInstance}, Base.CodegenParams, Cint),
    [ssg.entry], params, #=extern policy=# 1)

ccall(:jl_dump_native, Cvoid,
    (Ptr{Cvoid}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Csize_t),
    native_code, C_NULL, C_NULL, "exp_kernel.a", C_NULL, C_NULL, 0)

# These are better done beforehand at the LLVM level, but just to keep this example simple
symname = split(read(pipeline(`nm text.o`, `grep julia_exp_kernel`), String))[3]
run(`ar x exp_kernel.a`)
run(`objcopy -w --redefine-sym $symname=exp_kernel --globalize-symbol=exp_kernel text.o text2.o`)
run(`gcc -shared -o exp_kernel.so text2.o`)

@show (ccall((:exp_kernel, "./exp_kernel.so"), Float64, (Float64,), 1.0), exp_kernel(1.0))
