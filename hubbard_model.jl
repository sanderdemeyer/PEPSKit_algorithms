using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit

t = 1
U = 8

T = ComplexF64


pspace = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -1, 0) => 1, (1, 0, 1) => 1, (1, 0, -1) => 1, (0, 1, 0) => 1)
space0 = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -1, 0) => 1)
vspace = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -1, 0) => 1, (1, 0, 1) => 1, (1, 0, -1) => 1, (0, 1, 0) => 1)
# vspaces = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, -2, 0) => 1, (1, -1, 1) => 1, (1, -1, -1) => 1, (0, 0, 0) => 1, (2, 0, 2) => 1, (2, 0, 0) => 1, (1, 1, 1) => 1, (2, 0, -2) => 1, (1, 1, -1) => 1, (0, 2, 0) => 1)

pspace = Vect[FermionParity](0 => 1, 1 => 1)

hx = TensorMap(zeros, T, pspace^2 ← pspace^2)

# space0 = Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](0, -1, 0)
# block(hx, space0)
# block(hx, FermionParity(1) ⊗ U1Irrep(1)) .= [0 -t; -t 0]

# blocks(hx)

for (block, fusiontree) in zip(blocks(hx), fusiontrees(hx))
    println(block)
    println(fusiontree)
end