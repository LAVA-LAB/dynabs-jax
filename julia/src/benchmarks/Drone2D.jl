function Drone2D()

    # Set value of delta (how many time steps are grouped together)
    # Used to make the model fully actuated
    lump = 2

    # Discretization step size
    tau = 1.0

    # State transition matrix
    Ablock = Float64[1 tau; 0 1]
    I = Diagonal(ones(2))
    A = kron(I, Ablock)

    Bblock = Float64[tau^2 / 2; tau]
    I = Diagonal(ones(2))
    B = kron(I, Bblock)

    Q = Float64[0; 0; 0; 0]

    vars = String["x_pos", "x_vel", "y_pos", "y_vel"]

    noise_cov = Diagonal(ones(size(A)[1])) * 0.15
    
    # Set control limits
    uMin = Float64[-4, -4]
    uMax = Float64[4, 4]

    partition_boundary = Float64[-7 -3 -7 -3; 7 3 7 3] .* 2
    partition_size = Int32[7; 4; 7; 4] .* 2

    goal = Array[
        Float64[5 -3 5 -3; 7 3 7 3]
    ]

    unsafe = Array[
        Float64[-7 -3 1 -3; -1 3 3 3],
        Float64[3 -3 -7 -3; 7 3 -3 3]
    ]

    # Set model as dictionary
    model = Dict("A"=>A, 
                 "B"=>B, 
                 "Q"=>Q, 
                 "uMin"=>uMin,
                 "uMax"=>uMax,
                 "vars"=>vars, 
                 "noise_cov"=>noise_cov,
                 "partition_boundary"=>partition_boundary,
                 "partition_size"=>partition_size,
                 "goal"=>goal,
                 "unsafe"=>unsafe,
                 "lump"=>lump,
                 "tau"=>tau,)

    return model

end