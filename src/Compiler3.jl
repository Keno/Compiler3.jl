module Compiler3

    using Core.Compiler
    using Core.Compiler: MethodInstance, NativeInterpreter, CodeInfo
    using Base.Meta

    import Core.Compiler: InferenceParams, OptimizationParams, get_world_counter,
        get_inference_cache, InferenceResult, _methods_by_ftype, OptimizationState,
        CodeInstance, Const, widenconst, isconstType

    export StaticSubGraph

    include("extracting_interpreter.jl")

    abstract type FunctionGraph; end

    struct StaticSubGraph <: FunctionGraph
        code::Dict{MethodInstance, Any}
        instances::Vector{MethodInstance}
        entry::MethodInstance
    end
    entrypoint(fg::StaticSubGraph) = fg.entry

    mi_at_cursor(mi::MethodInstance) = mi

    has_codeinfo(ssg::StaticSubGraph, mi::MethodInstance) = haskey(ssg.code, mi)
    function get_codeinfo(ssg::StaticSubGraph, mi::MethodInstance)
        code = ssg.code[mi]
        ci = code.inferred
        isa(ci, Vector{UInt8}) && (ci = Core.Compiler._uncompressed_ir(code, ci))
        ci
    end

    function analyze(types::Type{<:Tuple})
        ei = ExtractingInterpreter()
        mi, result = infer_function(ei, types)
        ei, StaticSubGraph(ei.code, collect(keys(ei.code)), mi)
    end

    function gather_children(ei, ssg, mi)
        # Run inlining to convert calls to invoke, which are easier to analyze
        haskey(ssg.code, mi) || return Any[] # TODO: When does this happen?
        ci = get_codeinfo(ssg, mi)
        params = OptimizationParams()
        sv = OptimizationState(ssg.entry, ci, params, ei)
        sv.slottypes .= ci.slottypes
        nargs = Int(sv.nargs) - 1
        ir = Core.Compiler.run_passes(ci, nargs, sv)
        ret = Any[]
        for stmt in ir.stmts
            isexpr(stmt, :invoke) || continue
            push!(ret, stmt.args[1])
        end
        unique(ret)
    end

    function find_msgs(ei, mi)
        filter(x->x[1] == mi, ei.msgs)
    end

    function analyze_static(types::Type{<:Tuple})
        ei = ExtractingInterpreter()
        mi, result = infer_function(ei, types)
        ei, ssg = ei, StaticSubGraph(ei.code, collect(keys(ei.code)), mi)
        worklist = Any[(ssg.entry, [])]
        visited = Set{Any}(worklist)
        while !isempty(worklist)
            mi, stack = popfirst!(worklist)
            global cur_mi
            cur_mi = mi
            for msg in find_msgs(ei, mi)
                print("In function: ")
                Base.show_tuple_as_call(stdout, mi.def.name, mi.specTypes)
                println()
                printstyled("ERROR: ", color=:red)
                println(msg[3]);
                ci = get_codeinfo(ssg, mi)
                loc = ci.linetable[ci.codelocs[msg[2]]]
                fname = String(loc.file)
                if startswith(fname, "REPL[")
                    hp = Base.active_repl.interface.modes[1].hist
                    repl_id = parse(Int, fname[6:end-1])
                    repl_contents = hp.history[repl_id+hp.start_idx]
                    for (n, line) in enumerate(split(repl_contents, '\n'))
                        print(n == loc.line ? "=> " : "$n| ")
                        println(line)
                    end
                else
                    println("TODO: File content here")
                end
                println()
                for (i, old_mi) in enumerate(reverse(stack))
                    print("[$i] In ")
                    Base.show_tuple_as_call(stdout, old_mi.def.name, old_mi.specTypes)
                    println()
                end
                println()
            end
            children = gather_children(ei, ssg, mi)
            for child in children
                if !(child in visited)
                    push!(worklist, (child, [copy(stack); mi]))
                end
                push!(visited, mi)
            end
        end
    end

    function optimize!(ei, ssg::StaticSubGraph)
        ci = get_codeinfo(ssg, ssg.entry)
        params = OptimizationParams()
        sv = OptimizationState(ssg.entry, ci, params, ei)
        sv.slottypes[:] = ci.slottypes
        nargs = Int(sv.nargs) - 1
        Core.Compiler.run_passes(ci, nargs, sv)
    end
end
