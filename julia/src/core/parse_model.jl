function parse_model(base_model)
    #=
    Parse linear dynamical model
    =#

    # Partition size
    base_model["partition_boundary"] = Float32.(base_model["partition_boundary"])
    base_model["partition_size"] = Int8.(base_model["partition_size"])

    # Control limits
    base_model["uMin"] = Float32.(base_model["uMin"])
    base_model["uMax"] = Float32.(base_model["uMax"])

    base_model["n"] = size(base_model["A"])[1]

    # Make the model fully actuated
    model = make_fully_actuated(base_model)

    # Determine vertices of the control input space
    stacked = [[l,u] for (l,u) in zip(model["uMin"], model["uMax"])]
    matr = collect.(Iterators.product(stacked...))
    model["uVertices"] = hcat(matr...)'

    # Determine inverse A matrix
    model["A_inv"] = inv(model["A"])

    # Determine pseudo-inverse B matrix
    model["B_pinv"] = pinv(model["B"])

    # Retreive system dimensions
    model["p"] = size(model["B"])[2]

    return model
    
end

function make_fully_actuated(base_model)

    model = copy(base_model)

    if model["lump"] != 0
        dim = model["lump"]
    else
        dim = Int8.(ceil(size(model["A"])[1] / size(model["B"])[1]))
    end

    # Determine fully actuated system matrices and parameters
    model["A"] = base_model["A"] ^ dim
    model["B"] = Matrix( vcat([base_model["A"]^(dim-i) * base_model["B"] for i in 1:dim]'...)' )
    model["Q"] = sum([base_model["A"]^(dim-i) * base_model["Q"] for i in 1:dim])
    model["noise_cov"] = sum([base_model["A"]^(dim-i) * base_model["noise_cov"] * (base_model["A"]')^(dim-i) for i in 1:dim])

    # Redefine sampling time of model
    model["tau"] = base_model["tau"] * dim

    # Set control limits
    model["uMin"] = repeat(base_model["uMin"], dim)
    model["uMax"] = repeat(base_model["uMax"], dim)

    return model

end