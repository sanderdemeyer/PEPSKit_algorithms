function clebschgordan(a, b, c, d, e, f)
    return -1
end

physical_space = Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0,0,0) => 1, (1,1,1/2) => 1, (0,2,0) => 1)
A = Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((1,1,1/2) => 1)
odd = FermionParity(1)
even = FermionParity(0)
c⁺ = TensorMap(zeros, ComplexF64, physical_space ← physical_space ⊗ A)
for (s, f) in fusiontrees(c⁺)
    if f.uncoupled[1].sectors[1] == even && f.uncoupled[1].sectors[2] == Irrep[U₁](0) && f.uncoupled[1].sectors[3]== Irrep[SU₂](0)
        c⁺[s, f] .= 1 end
     if f.uncoupled[1].sectors[1] == odd && f.uncoupled[1].sectors[2] == Irrep[U₁](1) && f.uncoupled[1].sectors[3]== Irrep[SU₂](1/2)
        c⁺[s, f] .= -sqrt(2)*clebschgordan(1/2, 1/2, 1/2, -1/2, 0, 0) end
end

c = TensorMap(zeros, ComplexF64,  A ⊗ physical_space ← physical_space)
for (s, f) in fusiontrees(c)
    if f.uncoupled[1].sectors[1] == odd && f.uncoupled[1].sectors[2] == Irrep[U₁](1) && f.uncoupled[1].sectors[3]== Irrep[SU₂](1/2)
        c[s, f] .= 1 end
     if f.uncoupled[1].sectors[1] == even && f.uncoupled[1].sectors[2] == Irrep[U₁](2) && f.uncoupled[1].sectors[3]== Irrep[SU₂](0)
        c[s, f] .= -sqrt(2)*clebschgordan(1/2, 1/2, 1/2, -1/2, 0, 0) end
end

@tensor c⁺c⁻[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]


