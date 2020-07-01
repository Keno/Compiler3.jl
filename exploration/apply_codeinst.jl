using Revise
using Core: SimpleVector

includet("zygoteng.jl"); includet("pinn.jl"); using Compiler3; using Cthulhu
interp = ADInterpreter()
function hook end
params = Base.CodegenParams(;generic_context=hook)

bar() = "Hello World"
function foo()
    Base.inferencebarrier(bar)()
end

function _apply_codeinst(ci::CodeInstance, f, args::Vector{Any})
    ccall(ci.invoke, Any, (Any, Ptr{Any}, Csize_t, Any), f, args, length(args), ci)
end

function hook(args...)
    mi, result = Compiler3.infer_function(interp, Base.typesof(args...))
    cursor = ADCursor(0, mi)
    ci = Compiler3.get_codeinstance(graph, cursor)
    if ci.invoke == C_NULL
        Compiler3.jit_compile(graph, cursor, params)
    end
    return _apply_codeinst(interp.code[0][mi], args[1], Any[args[2:end]...])
end

mi, result = Compiler3.infer_function(interp, Tuple{typeof(foo)})
graph = ADGraph(interp.code, interp.msgs, mi)

Compiler3.jit_compile(graph, entrypoint(graph), params)
_apply_codeinst(interp.code[0][mi], foo, Any[])
