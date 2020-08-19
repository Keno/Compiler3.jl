# LLVM IR generation


## method compilation tracer

# this functionality is used to detect recursion, and functions that shouldn't be called.
# it is a hack, and should disappear over time. don't add new features to it.

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(job::CompilerJob, call_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in call_stack
        method = method_instance.def
        if method.name === :overdub && isdefined(method, :generator)
            # The inline frames are maintained by the dwarf based backtrace, but here we only have the
            # calls to overdub directly, the backtrace therefore is collapsed and we have to
            # lookup the overdubbed function, but only if we likely are using the generated variant.
            actual_sig = Tuple{method_instance.specTypes.parameters[3:end]...}
            m = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), actual_sig, typemax(UInt))
            method = m.func::Method
        end
        frame = StackTraces.StackFrame(method.name, method.file, method.line)
        pushfirst!(bt, frame)
    end
    bt
end

# NOTE: we use an exception to be able to display a stack trace using the logging framework
struct MethodSubstitutionWarning <: Exception
    original::Method
    substitute::Method
end
Base.showerror(io::IO, err::MethodSubstitutionWarning) =
    print(io, "You called $(err.original), maybe you intended to call $(err.substitute) instead?")
const method_substitution_whitelist = [:hypot, :exp]

mutable struct MethodCompileTracer
    job::CompilerJob
    call_stack::Vector{Core.MethodInstance}
    last_method_instance::Union{Nothing,Core.MethodInstance}

    MethodCompileTracer(job, start) = new(job, Core.MethodInstance[start])
    MethodCompileTracer(job) = new(job, Core.MethodInstance[])
end

function Base.push!(tracer::MethodCompileTracer, method_instance)
    push!(tracer.call_stack, method_instance)

    # check for Base functions that exist in the GPU package
    # FIXME: this might be too coarse
    method = method_instance.def
    if Base.moduleroot(method.module) == Base &&
        isdefined(runtime_module(tracer.job), method_instance.def.name) &&
        !in(method_instance.def.name, method_substitution_whitelist)
        substitute_function = getfield(runtime_module(tracer.job), method.name)
        tt = Tuple{method_instance.specTypes.parameters[2:end]...}
        if hasmethod(substitute_function, tt)
            method′ = which(substitute_function, tt)
            if method′.module == runtime_module(tracer.job)
                @warn "calls to Base intrinsics might be GPU incompatible" exception=(MethodSubstitutionWarning(method, method′), backtrace(tracer.job, tracer.call_stack))
            end
        end
    end
end

function Base.pop!(tracer::MethodCompileTracer, method_instance)
    @compiler_assert last(tracer.call_stack) == method_instance tracer.job
    tracer.last_method_instance = pop!(tracer.call_stack)
end

Base.last(tracer::MethodCompileTracer) = tracer.last_method_instance


## Julia compiler integration

### cache

using Core.Compiler: CodeInstance, MethodInstance

struct GPUCodeCache
    dict::Dict{MethodInstance,Vector{CodeInstance}}
    GPUCodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
end

function Core.Compiler.setindex!(cache::GPUCodeCache, ci::CodeInstance, mi::MethodInstance)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

const GPU_CI_CACHE = GPUCodeCache()

### interpreter

using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams

struct GPUInterpreter <: AbstractInterpreter
    # Cache of inference results for this particular interpreter
    cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    function GPUInterpreter(world::UInt)
        @assert world <= Base.get_world_counter()

        return new(
            # Initially empty cache
            Vector{InferenceResult}(),

            # world age counter
            world,

            # parameters for inference and optimization
            InferenceParams(),
            OptimizationParams(),
        )
    end
end

# Quickly and easily satisfy the AbstractInterpreter API contract
Core.Compiler.get_world_counter(ni::GPUInterpreter) = ni.world
Core.Compiler.get_inference_cache(ni::GPUInterpreter) = ni.cache
Core.Compiler.InferenceParams(ni::GPUInterpreter) = ni.inf_params
Core.Compiler.OptimizationParams(ni::GPUInterpreter) = ni.opt_params
Core.Compiler.may_optimize(ni::GPUInterpreter) = true
Core.Compiler.may_compress(ni::GPUInterpreter) = true
Core.Compiler.may_discard_trees(ni::GPUInterpreter) = true
Core.Compiler.add_remark!(ni::GPUInterpreter, sv::InferenceState, msg) = nothing # TODO

### world view of the cache

using Core.Compiler: WorldView

function Core.Compiler.haskey(wvc::WorldView{GPUCodeCache}, mi::MethodInstance)
    Core.Compiler.get(wvc, mi, nothing) !== nothing
end

function Core.Compiler.get(wvc::WorldView{GPUCodeCache}, mi::MethodInstance, default)
    cache = wvc.cache
    for ci in get!(cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            # TODO: if (code && (code == jl_nothing || jl_ir_flag_inferred((jl_array_t*)code)))
            return ci
        end
    end

    return default
end

function Core.Compiler.getindex(wvc::WorldView{GPUCodeCache}, mi::MethodInstance)
    r = Core.Compiler.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

Core.Compiler.setindex!(wvc::WorldView{GPUCodeCache}, ci::CodeInstance, mi::MethodInstance) =
    Core.Compiler.setindex!(wvc.cache, ci, mi)

### codegen/interence integration

Core.Compiler.code_cache(ni::GPUInterpreter) = WorldView(GPU_CI_CACHE, ni.world)

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(ni::GPUInterpreter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(ni::GPUInterpreter, mi::MethodInstance) = nothing

function gpu_ci_cache_lookup(mi, min_world, max_world)
    wvc = WorldView(GPU_CI_CACHE, min_world, max_world)
    if !Core.Compiler.haskey(wvc, mi)
        interp = GPUInterpreter(min_world)
        src = Core.Compiler.typeinf_ext_toplevel(interp, mi)
        # inference populates the cache, so we don't need to jl_get_method_inferred
        @assert Core.Compiler.haskey(wvc, mi)

        # if src is rettyp_const, the codeinfo won't cache ci.inferred
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        ci = Core.Compiler.getindex(wvc, mi)
        if ci !== nothing && ci.inferred === nothing
            ci.inferred = src
        end
    end
    return Core.Compiler.getindex(wvc, mi)
end

### external interface

function compile_method_instance(job::CompilerJob, method_instance::Core.MethodInstance, world)
    # set-up the compiler interface
    tracer = MethodCompileTracer(job, method_instance)
    hook_emit_function(method_instance, code) = push!(tracer, method_instance)
    hook_emitted_function(method_instance, code) = pop!(tracer, method_instance)
    debug_info_kind = if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
    if job.target isa PTXCompilerTarget # && driver_version(job.target) < v"10.2"
        # LLVM's debug info crashes older CUDA assemblers
        # FIXME: this was supposed to be fixed on 10.2
        @debug "Incompatibility detected between CUDA and LLVM 8.0+; disabling debug info emission" maxlog=1
        debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
    end
    params = Base.CodegenParams(;
                    track_allocations  = false,
                    code_coverage      = false,
                    static_alloc       = false,
                    prefer_specsig     = true,
                    emit_function      = hook_emit_function,
                    emitted_function   = hook_emitted_function,
                    gnu_pubnames       = false,
                    debug_info_kind    = Cint(debug_info_kind),
                    lookup             = @cfunction(gpu_ci_cache_lookup, Any, (Any, UInt, UInt)))

    # generate IR
    native_code = ccall(:jl_create_native, Ptr{Cvoid},
                        (Vector{Core.MethodInstance}, Base.CodegenParams, Cint),
                        [method_instance], params, #=extern policy=# 1)
    @assert native_code != C_NULL
    llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                         (Ptr{Cvoid},), native_code)
    @assert llvm_mod_ref != C_NULL
    llvm_mod = LLVM.Module(llvm_mod_ref)

    # get the top-level code
    code = gpu_ci_cache_lookup(method_instance, world, world)

    # get the top-level function index
    llvm_func_idx = Ref{Int32}(-1)
    llvm_specfunc_idx = Ref{Int32}(-1)
    ccall(:jl_breakpoint, Nothing, ())
    ccall(:jl_get_function_id, Nothing,
          (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
          native_code, code, llvm_func_idx, llvm_specfunc_idx)
    @assert llvm_func_idx[] != -1
    @assert llvm_specfunc_idx[] != -1

    # get the top-level function)
    llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                     (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
    @assert llvm_func_ref != C_NULL
    llvm_func = LLVM.Function(llvm_func_ref)
    llvm_specfunc_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                         (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
    @assert llvm_specfunc_ref != C_NULL
    llvm_specfunc = LLVM.Function(llvm_specfunc_ref)

    # configure the module
    triple!(llvm_mod, llvm_triple(job.target))
    if llvm_datalayout(job.target) !== nothing
        datalayout!(llvm_mod, llvm_datalayout(job.target))
    end

    return llvm_specfunc, llvm_mod
end

function irgen(job::CompilerJob, method_instance::Core.MethodInstance, world)
    entry, mod = @timeit_debug to "emission" compile_method_instance(job, method_instance, world)

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            # only occurs in debug builds
            delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, JuliaContext()))

            if VERSION < v"1.5.0-DEV.393"
                # make function names safe for ptxas
                llvmfn = LLVM.name(llvmf)
                if !isdeclaration(llvmf)
                    llvmfn′ = safe_name(llvmfn)
                    if llvmfn != llvmfn′
                        LLVM.name!(llvmf, llvmfn′)
                        llvmfn = llvmfn′
                    end
                end
            end

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    # target-specific processing
    process_module!(job, mod)

    # sanitize function names
    # FIXME: Julia should do this, but apparently fails (see maleadt/LLVM.jl#201)
    for f in functions(mod)
        LLVM.isintrinsic(f) && continue
        llvmfn = LLVM.name(f)
        startswith(llvmfn, "julia.") && continue # Julia intrinsics
        startswith(llvmfn, "llvm.") && continue # unofficial LLVM intrinsics
        llvmfn′ = safe_name(llvmfn)
        if llvmfn != llvmfn′
            @assert !haskey(functions(mod), llvmfn′)
            LLVM.name!(f, llvmfn′)
        end
    end

    # rename the entry point
    if job.source.name !== nothing
        LLVM.name!(entry, safe_name(string("julia_", job.source.name)))
    end

    # promote entry-points to kernels and mangle its name
    if job.source.kernel
        entry = promote_kernel!(job, mod, entry)
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        internalize!(pm, [LLVM.name(entry)])

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)

        # NOTE: if an optimization is missing, try scheduling an entirely new optimization
        # to see which passes need to be added to the target-specific list
        #     LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
        #     ModulePassManager() do pm
        #         add_library_info!(pm, triple(mod))
        #         add_transform_info!(pm, tm)
        #         PassManagerBuilder() do pmb
        #             populate!(pm, pmb)
        #         end
        #         run!(pm, mod)
        #     end
    end

    return mod, entry
end


## name mangling

# we generate function names that look like C++ functions, because many NVIDIA tools
# support them, e.g., grouping different instantiations of the same kernel together.

function mangle_param(t, substitutions)
    t == Nothing && return "v"

    if isa(t, DataType) || isa(t, Core.Function)
        tn = safe_name(t)

        # handle substitutions
        sub = findfirst(isequal(tn), substitutions)
        if sub === nothing
            str = "$(length(tn))$tn"
            push!(substitutions, tn)
        elseif sub == 1
            str = "S_"
        else
            str = "S$(sub-2)_"
        end

        # encode typevars as template parameters
        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t, substitutions)
            end
            str *= "E"
        end

        str
    elseif isa(t, Integer)
        "Li$(t)E"
    else
        tn = safe_name(t)
        "$(length(tn))$tn"
    end
end

function mangle_call(f, tt)
    fn = safe_name(f)
    str = "_Z$(length(fn))$fn"

    substitutions = String[]
    for t in tt.parameters
        str *= mangle_param(t, substitutions)
    end

    return str
end

# make names safe for ptxas
safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9_]"=>"_")
safe_name(f::Union{Core.Function,DataType}) = safe_name(String(nameof(f)))
safe_name(f::LLVM.Function) = safe_name(LLVM.name(f))
safe_name(x) = safe_name(repr(x))


## exception handling

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "lower throw" begin

    throw_functions = [
        # unsupported runtime functions that are used to throw specific exceptions
        "jl_throw"                      => "exception",
        "jl_error"                      => "error",
        "jl_too_few_args"               => "too few arguments exception",
        "jl_too_many_args"              => "too many arguments exception",
        "jl_type_error"                 => "type error",
        "jl_type_error_rt"              => "type error",
        "jl_undefined_var_error"        => "undefined variable error",
        "jl_bounds_error"               => "bounds error",
        "jl_bounds_error_v"             => "bounds error",
        "jl_bounds_error_int"           => "bounds error",
        "jl_bounds_error_tuple_int"     => "bounds error",
        "jl_bounds_error_unboxed_int"   => "bounds error",
        "jl_bounds_error_ints"          => "bounds error",
        "jl_eof_error"                  => "EOF error",
        # Julia-level exceptions that use unsupported inputs like interpolated strings
        r"julia_throw_exp_domainerror_\d+"      => "DomainError",
        r"julia_throw_complex_domainerror_\d+"  => "DomainError"
    ]

    for f in functions(mod)
        fn = LLVM.name(f)
        for (throw_fn, name) in throw_functions
            occursin(throw_fn, fn) || continue

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                let builder = Builder(JuliaContext())
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                    dispose(builder)
                end

                # remove the call
                call_args = collect(operands(call))[1:end-1] # last arg is function itself
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(operands(cast))
                        isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                        unsafe_delete!(LLVM.parent(arg), arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(uses(f)) job
            break
         end
     end

    end
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` will be emitted, On
# debug level 1, the exception name will be printed, and on debug level 2 the individual
# stack frames (as recovered from the LLVM debug information) will be printed as well.
function emit_exception!(builder, name, inst)
    job = current_job::CompilerJob
    bb = position(builder)
    fun = LLVM.parent(bb)
    mod = LLVM.parent(fun)

    # report the exception
    if Base.JLOptions().debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if Base.JLOptions().debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if Base.JLOptions().debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(rt.llvm_types[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(rt.llvm_types[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception
    call!(builder, Runtime.get(:signal_exception))

    emit_trap!(job, builder, mod, inst)
end

function emit_trap!(job::CompilerJob, builder, mod, inst)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(JuliaContext())))
    end
    call!(builder, trap)
end


## kernel promotion

# promote a function to a kernel
function promote_kernel!(job::CompilerJob, mod::LLVM.Module, kernel::LLVM.Function)
    # pass non-opaque pointer arguments by value (this improves performance,
    # and is mandated by certain back-ends like SPIR-V). only do so for values
    # that aren't a Julia pointer, so we ca still pass those directly.
    kernel_ft = eltype(llvmtype(kernel)::LLVM.PointerType)::LLVM.FunctionType
    kernel_sig = Base.signature_type(job.source.f, job.source.tt)::Type
    kernel_types = filter(dt->!isghosttype(dt) &&
                              (VERSION < v"1.5.0-DEV.581" || !Core.Compiler.isconstType(dt)),
                          [kernel_sig.parameters...])
    @compiler_assert length(kernel_types) == length(parameters(kernel_ft)) job
    for (i, (param_ft,arg_typ)) in enumerate(zip(parameters(kernel_ft), kernel_types))
        if param_ft isa LLVM.PointerType && issized(eltype(param_ft)) &&
           !(arg_typ <: Ptr) && !(VERSION >= v"1.5-" && arg_typ <: Core.LLVMPtr)
            push!(parameter_attributes(kernel, i), EnumAttribute("byval"))
        end
    end

    # target-specific processing
    kernel = process_kernel!(job, mod, kernel)

    return kernel
end
