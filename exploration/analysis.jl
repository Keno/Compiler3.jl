using Compiler3
using Compiler3: StaticSubGraph

function bar()
    sin = eval(:sin)
    sin(1)
end
foo() = bar()

ei, ssg = Compiler3.analyze_static(Tuple{typeof(foo)})
ssg.entry
