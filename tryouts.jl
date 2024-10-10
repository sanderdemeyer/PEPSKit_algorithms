using TensorKit, KrylovKit, OptimKit
using PEPSKit


dims = (3, 4, 5, 2, 6, 7)  # Example dimensions for each of the 6 indices

# Create a random tensor with the specified dimensions
A = TensorMap(randn, ℂ^3 ⊗ ℂ^2 ⊗ ℂ^4, ℂ^3 ⊗ ℂ^2 ⊗ ℂ^4)

(U, S, V) = PEPSKit.tsvd(A)

@tensor A_new[-1 -2 -3; -4 -5 -6] := U[-1 -2 -3; 1] * S[1; 2] * V[2; -4 -5 -6]
norm(A-A_new)
@tensor A_new[-1 -2 -3; -4 -5 -6] := U[-1 -2 -3; 1] * S[1; 2] * conj(V[-4 -5 -6; -2])



print(a)
T = ComplexF64
J=1
h=1
physical_space = ComplexSpace(2)
lattice = fill(physical_space, 3, 4)
σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
hzz = nearest_neighbour_hamiltonian(lattice, -J * σz ⊗ σz)
local_operator = repeat(
    LocalOperator(lattice, hzz.terms..., (CartesianIndex(1, 1),) => -J * h * σx),
    unitcell...,
)

terms = []
for I in eachindex(IndexCartesian(), lattice)
    println(I)
    J1 = I + CartesianIndex(1, 0)
    J2 = I + CartesianIndex(0, 1)
    push!(terms, (I, J1) => h)
    push!(terms, (I, J2) => h)
end


tr = map(((inds,operator),)-> (inds => 5*operator), [(1, 2) (2,4) (3,6)])