function compute_enabled_actions(As::Vector{Matrix{Float64}}, bs::Vector{Vector{Float64}}, Rv::Vector{Matrix{Float64}})
    #=
    Determine which action is enabled in which state
    =#

    # Rv_flat = hcat(Rv...)

    enabled_actions = zeros(size(As)[1], size(As)[1])::Matrix{Float64}

    for (a,(A,b)) in enumerate(zip(As,bs))
        enabled_actions[a,:] = @time compute_single_action(A,b,Rv)::Array{Bool}
    end

    return enabled_actions

end

function compute_single_action(A::Matrix{Float64}, b::Vector{Float64}, Rv::Array{Matrix{Float64}, 4})

    return [all(A * r .<= b) for r in Rv]::Array{Bool}

end