# module abstraction

using LinearAlgebra
using LazySets
using LazyGrids
using Polyhedra
using ProgressBars

include("benchmarks/Drone2D.jl")
include("core/parse_model.jl")
include("core/partition.jl")
include("core/enabled_actions.jl")
include("core/utils.jl")

t = time()
base_model = Drone2D()
model = parse_model(base_model)
println("- Benchmark defined and parsed ($(timediff(t)) sec.)")

t = time()
centers, R, Rv, R_G, R_U = create_partition(model["partition_size"],
                                    model["partition_boundary"],
                                    model["goal"],
                                    model["unsafe"])
println("- Partition created ($(timediff(t)) sec.)")

t = time()
backreachsets, As, bs, target_points = create_actions(centers, model)
println("- Actions created ($(timediff(t)) sec.)")



s = Int32(14*8*14*8)
As2 = Array{Matrix{Float64}}(undef, s);
for (i,A) in enumerate(As)
    As2[i] = A 
end
bs2 = Array{Vector{Float64}}(undef, s);
for (i,b) in enumerate(bs)
    bs2[i] = b
end
Rv2 = Array{Matrix{Float64}}(undef, s);
for (i,r) in enumerate(Rv)
    Rv2[i] = r
end

t = time()
enabled_actions = compute_enabled_actions(As2, bs2, Rv2)
println("- Enabled actions computed ($(timediff(t)) sec.)")


# end