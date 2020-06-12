using ChainRules
using OffsetArrays
using Cthulhu

struct Pullback
    data::Any
end
(::Pullback)(args...) = Base.inferencebarrier(error)("MAGIC")

function pullback(f, args...)
    error("MAGIC")
end

function gradient(f, args...)
    y, back = pullback(f, args...)
    return Base.tail(back(one(y)))
end

Base.adjoint(f::Function) = x -> gradient(f, x)[1]

using Core: MethodInstance, CodeInstance
import Core.Compiler: InferenceParams, OptimizationParams, get_world_counter,
    get_inference_cache, code_cache,
    WorldView, lock_mi_inference, unlock_mi_inference, InferenceState,
    AbstractInterpreter, NativeInterpreter

using Compiler3
import Compiler3: entrypoint, mi_at_cursor, FunctionGraph

struct ADCursor
    level::Int
    mi::MethodInstance
end
mi_at_cursor(c::ADCursor) = c.mi
Cthulhu.get_cursor(c::ADCursor, callinfo::Cthulhu.PullbackCallInfo) = ADCursor(c.level+1, Cthulhu.get_mi(callinfo.mi))
Cthulhu.get_cursor(c::ADCursor, cs::Cthulhu.Callsite) = Cthulhu.get_cursor(c, cs.info)
Cthulhu.get_cursor(c::ADCursor, callinfo) = ADCursor(c.level, Cthulhu.get_mi(callinfo))

struct ADGraph <: FunctionGraph
    code::OffsetVector{Dict{MethodInstance, Any}}
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
    entry_mi::MethodInstance
end
entrypoint(graph::ADGraph) = ADCursor(0, graph.entry_mi)
Compiler3.has_codeinfo(graph::ADGraph, cursor::ADCursor) =
    lastindex(graph.code) >= cursor.level && haskey(graph.code[cursor.level], cursor.mi)
function Compiler3.get_codeinfo(graph::ADGraph, cursor::ADCursor)
    code = graph.code[cursor.level][cursor.mi]
    ci = code.inferred
    isa(ci, Vector{UInt8}) && (ci = Core.Compiler._uncompressed_ir(code, ci))
    ci
end

struct RRuleCallInfo <: Cthulhu.CallInfo
    rrule_mi::MethodInstance
    rrule_rt::Any
    ci::Cthulhu.CallInfo
    new_level::Int
end

function Cthulhu.callinfo(fg::ADGraph, c::ADCursor, sig, rt, max=-1; params=Cthulhu.current_params())
    real_ci = Cthulhu.callinfo(fg, c.mi, sig, rt, max; params=params)
    if c.level >= 1
        tt = Tuple{typeof(ChainRules.rrule), sig.parameters...}
        # Find all methods that are applicable to these types
        mthds = _methods_by_ftype(tt, -1, typemax(UInt))
        if mthds === false || length(mthds) != 1
            return Cthulhu.FailedCallInfo(sig, rt)
        end

        mtypes, msp, m = mthds[1]

        # Grab the appropriate method instance for these types
        mi = Core.Compiler.specialize_method(m, mtypes, msp)
        nc = ADCursor(c.level-1, mi)

        if Compiler3.has_codeinfo(fg, nc)
            rrule_rt = Compiler3.get_codeinfo(fg, nc).rettype
            if rrule_rt != Const(nothing)
                rrule_rt = Core.Compiler.getfield_tfunc(rrule_rt, Const(1))
                return RRuleCallInfo(mi, rrule_rt, real_ci, c.level-1)
            end
        end
    end
    return real_ci
end

function Base.show(io::IO, rrule::RRuleCallInfo)
    print(io, "= rrule < ")
    Cthulhu.show_callinfo(io, rrule.ci)
    print(io, " >::")
    print(io, string(rrule.rrule_rt))
end

function Compiler3.find_msgs(graph::ADGraph, cursor::ADCursor)
    filter(x->x[1] == cursor.level && x[2] == cursor.mi, graph.msgs)
end

function Cthulhu.get_cursor(cursor::ADCursor, rr::RRuleCallInfo)
    ADCursor(cursor.level - 1, rr.rrule_mi)
end

struct ADInterpreter <: AbstractInterpreter
    # This cache is stratified by AD nesting level. Depending on the
    # nesting level of the derivative, The AD primitives may behave
    # differently.
    # Level 0 == Straightline Code, no AD
    # Level 1 == Gradients
    # Level 2 == Seconds Derivatives
    # and so on
    code::OffsetVector{Dict{MethodInstance, Any}}
    native_interpreter::NativeInterpreter
    current_level::Int
    msgs::Vector{Tuple{Int, MethodInstance, Int, String}}
end
raise_level(interp::ADInterpreter) = ADInterpreter(interp.code, interp.native_interpreter, interp.current_level + 1, interp.msgs)
lower_level(interp::ADInterpreter) = ADInterpreter(interp.code, interp.native_interpreter, interp.current_level - 1, interp.msgs)

function Core.Compiler.is_same_frame(interp::ADInterpreter, linfo::MethodInstance, frame::InferenceState)
    linfo === frame.linfo || return false
    return interp.current_level === frame.interp.current_level
end

ADInterpreter() = ADInterpreter(
    OffsetVector([Dict{MethodInstance, Any}(), Dict{MethodInstance, Any}()], 0:1),
    NativeInterpreter(),
    0,
    Vector{Tuple{Int, MethodInstance, Int, String}}()
)

ADInterpreter(fg::ADGraph, level) =
    ADInterpreter(fg.code, NativeInterpreter(), level, fg.msgs)

struct AbstractPullback
    f::Any
    argtypes::Vector{Any}
end

using Core.Compiler: Const, isconstType, argtypes_to_type, tuple_tfunc, Const,
    getfield_tfunc, _methods_by_ftype, VarTable
import Core.Compiler: abstract_call_gf_by_type, abstract_call, widenconst
widenconst(ap::AbstractPullback) = Pullback
function abstract_call_gf_by_type(interp::ADInterpreter, @nospecialize(f), argtypes::Vector{Any}, @nospecialize(atype), sv::InferenceState, max_methods = InferenceParams(interp).MAX_METHODS)
    # Check if this is `pullback`
    @show f
    if f === pullback
        inner_argtypes = argtypes[2:end]
        ft = inner_argtypes[1]
        f = nothing
        if isa(ft, Const)
            f = ft.val
        elseif isconstType(ft)
            f = ft.parameters[1]
        elseif isa(ft, DataType) && isdefined(ft, :instance)
            f = ft.instance
        end
        @show ("inner", f)
        # TODO: It would be probably be better to pre-infer the pullback during
        # the forward mode pass
        rt = abstract_call_gf_by_type(raise_level(interp), f, inner_argtypes, argtypes_to_type(inner_argtypes), sv, max_methods)
        @show rt
        rt2 = tuple_tfunc(Any[rt, AbstractPullback(f, argtypes)])
        @show rt2
        return rt2
    end
    # Check if there is a rrule for this function
    if f !== ChainRules.rrule && interp.current_level != 0
        rrule_argtypes = Any[Const(ChainRules.rrule); argtypes]
        rrule_atype = argtypes_to_type(rrule_argtypes)
        # In general we want the forward type of an rrule'd function to match
        # what the function itself would have returned, but let's support this
        # not being the case.
        @show ("rrule", interp.current_level, argtypes[1], argtypes[2:end])
        rt = abstract_call_gf_by_type(lower_level(interp), ChainRules.rrule, rrule_argtypes, rrule_atype, sv, -1)
        @show ("rrule", argtypes[1], rt)
        if rt != Const(nothing)
            return getfield_tfunc(rt, Const(1))
        end
    end
    invoke(abstract_call_gf_by_type,
        Tuple{AbstractInterpreter, Any, Vector{Any}, Any, InferenceState, Any},
        interp, f, argtypes, atype, sv, max_methods)
end

function adjoint_type(@nospecialize(typ))
    if isbitstype(typ)
        return typ
    elseif typ <: Array
        return typ
    elseif typ <: Tuple
        return Tuple{(adjoint_type(x) for x in typ.parameters)...}
    elseif isa(typ, Union)
        return Union{adjoint_type(typ.a), adjoint_type(typ.b)}
    elseif isa(typ, UnionAll)
        return Base.rewrap_unionall(adjoint_type(Base.unwrap_unionall(typ)), typ)
    elseif typ.abstract
        return Any
    else
        @show typ
        return NamedTuple{fieldnames(typ), Tuple{fieldtypes(typ)...}}
    end
end

function abstract_call_pullback(pullback::AbstractPullback, pullback_args::Vector{Any}, sv::InferenceState)
    # For now, assume the pullbacks exactly match the types of the arguments. In
    # The future, we will do the full reverse inference thing here.
    tuple_tfunc(Any[adjoint_type(widenconst(x)) for x in pullback.argtypes[2:end]])
end

function abstract_call(interp::ADInterpreter, fargs::Union{Nothing,Vector{Any}}, argtypes::Vector{Any},
        vtypes::VarTable, sv::InferenceState, max_methods = InferenceParams(interp).MAX_METHODS)
    if isa(argtypes[1], AbstractPullback)
        return abstract_call_pullback(argtypes[1], argtypes, sv)
    end
    invoke(abstract_call, Tuple{AbstractInterpreter, Union{Nothing, Vector{Any}}, Vector{Any}, VarTable, InferenceState, Any},
        interp, fargs, argtypes, vtypes, sv, max_methods)
end

InferenceParams(ei::ADInterpreter) = InferenceParams(ei.native_interpreter)
OptimizationParams(ei::ADInterpreter) = OptimizationParams(ei.native_interpreter)
get_world_counter(ei::ADInterpreter) = get_world_counter(ei.native_interpreter)
get_inference_cache(ei::ADInterpreter) = get_inference_cache(ei.native_interpreter)

# No need to do any locking since we're not putting our results into the runtime cache
lock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing
unlock_mi_inference(ei::ADInterpreter, mi::MethodInstance) = nothing

function code_cache(ei::ADInterpreter)
    while ei.current_level > lastindex(ei.code)
        push!(ei.code, Dict{MethodInstance, Any}())
    end
    ei.code[ei.current_level]
end
Core.Compiler.may_optimize(ei::ADInterpreter) = false
Core.Compiler.may_compress(ei::ADInterpreter) = false
Core.Compiler.may_discard_trees(ei::ADInterpreter) = false

function Core.Compiler.mark_dynamic!(ei::ADInterpreter, sv::InferenceState, msg)
    push!(ei.msgs, (ei.current_level, sv.linfo, sv.currpc, msg))
end
