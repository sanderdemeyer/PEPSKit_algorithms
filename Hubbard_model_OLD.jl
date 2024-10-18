using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit

include("Hubbard_tensors.jl")
include("hamiltonians.jl")

t = 1
U = 8

T = ComplexF64

function U1_charges(P, Q)
    return (-P, Q-P, 2*Q-P)
end

charges = U1_charges(1, 1)

function get_spaces(symmetry)
    if symmetry == "U1"
        pspace = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, charges[1], 0) => 1, (1, charges[2], 1) => 1, (1, charges[2], -1) => 1, (0, charges[3], 0) => 1)
        pspace = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, charges[1], 0) => 1, (1, charges[2], 1) => 1, (1, charges[2], -1) => 1, (0, charges[3], 0) => 1)
    else
        pspace = Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, charges[1], 0) => 1, (1, charges[2], 1/2) => 2, (0, charges[3], 0) => 1)
        plus_space = Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((1, charges[2], 1 // 2) => 1)
    end

vspace = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -1, 0) => 1, (1, 0, 1) => 1, (1, 0, -1) => 1, (0, 1, 0) => 1)
# vspaces = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -2, 0) => 1, (1, -1, 1) => 1, (1, -1, -1) => 1, (0, 0, 0) => 1, (2, 0, 2) => 1, (2, 0, 0) => 1, (1, 1, 1) => 1, (2, 0, -2) => 1, (1, 1, -1) => 1, (0, 2, 0) => 1)



hx = TensorMap(zeros, T, pspace^2 ← pspace^2)

# space0 = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](0, -1, 0)
# block(hx, space0)
# block(hx, FermionParity(1) ⊗ U1Irrep(1)) .= [0 -t; -t 0]

# blocks(hx)

for (block, fusiontree) in zip(blocks(hx), fusiontrees(hx))
    println(block)
    println(fusiontree)
end