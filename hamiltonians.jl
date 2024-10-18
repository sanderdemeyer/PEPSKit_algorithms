function nearest_neighbour_hamiltonian(
    lattice::InfiniteSquare, twosite::AbstractTensorMap{S,2,2}, onesite::AbstractTensorMap{S,1,1}
) where {S}
    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        J1 = I + CartesianIndex(1, 0)
        J2 = I + CartesianIndex(0, 1)
        push!(terms, (I, J1) => twosite)
        push!(terms, (I, J2) => twosite)
        push!(terms, I => onesite)
    end
    return LocalOperator(lattice, terms...)
end
