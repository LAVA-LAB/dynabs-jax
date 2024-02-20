function create_partition(number_per_dim, partition_boundary, goal, unsafe)

    cell_width = (partition_boundary[2,:] - partition_boundary[1,:]) ./ number_per_dim

    lb_center = partition_boundary[1,:] + cell_width * 0.5
    ub_center = partition_boundary[2,:] - cell_width * 0.5

    centers = define_grid(lb_center, ub_center, number_per_dim)
    
    # Use lazysets to generate regions in this 
    R = [Hyperrectangle(collect(center), 0.5 * cell_width) for center in zip(centers...)]
    Rv = [hcat(vertices_list(r)...) for r in R]

    EPS = 1e-5
    goal_sets = [Hyperrectangle((g[2,:]+g[1,:])/2, (g[2,:]-g[1,:])/2) for g in goal];
    unsafe_sets = [Hyperrectangle((u[2,:]+u[1,:])/2, (u[2,:]-u[1,:] .-0.1)/2) for u in unsafe];

    R_G = [[any(issubset(r, g) for g in goal_sets)] for r in R];
    R_U = [[any(!isdisjoint(r, u) for u in unsafe_sets)] for r in R];

    return centers, R, Rv, R_G, R_U

end

function define_grid(low, high, size)

    points = [LinRange(l, h, s) for (l,h,s) in zip(low, high, size)]
    grid = ndgrid(points...)

    return grid

end

function create_actions(target_points, model)

    A_inv = model["A_inv"]
    B = model["B"]
    Q = model["Q"]
    Uv = model["uVertices"]

    backreachsets = [compute_backreachset(d, A_inv, B, Q, Uv) for d in zip(target_points...)]

    As = [Matrix(hcat([hs.a for hs in constraints_list(S)]...)') for S in backreachsets]
    bs = [[hs.b for hs in constraints_list(S)] for S in backreachsets]

    return backreachsets, As, bs, target_points

end

function compute_backreachset(d, A_inv, B, Q, Uv)
    #=
    Compute the backward reachable set for the given target point 'd'.
    =#

    inner = d .- (B * Uv') .- Q
    vertices = A_inv * inner

    # Create backward reachable set as polytope
    backreachset = VPolytope(vertices)

    return backreachset

end