function timediff(start_time::Float64, digits::Int64=3)
    elapsed = round(time()-start_time, sigdigits=digits)

    return elapsed
end