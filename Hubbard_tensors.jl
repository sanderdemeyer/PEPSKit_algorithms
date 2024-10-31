function SymSpace(P,Q,spin)
    if spin
        I = fℤ₂ ⊠ U1Irrep ⊠ U1Irrep
        Ps = Vect[I]((0, -P, 0) => 1, (1, Q-P, 1) => 1, (1, Q-P, -1) => 1, (0, 2*Q-P, 0) => 1)
    else
        I = fℤ₂ ⊠ U1Irrep ⊠ SU2Irrep 
        Ps = Vect[I]((0, -P, 0) => 1, (1, Q-P, 1 // 2) => 1, (0, 2*Q-P, 0) => 1)
    end

    return I, Ps
end

function ASymSpace()
    I = fℤ₂
    Ps = Vect[I]((0) => 2, (1) => 2)
    return I, Ps
end    

function ASym_Hopping()
    I, Ps = ASymSpace()
    Vodd = Vect[I]((1) => 1)

    c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
    blocks(c⁺u)[I((1))] .= [1 0; 0 0]
    blocks(c⁺u)[I((0))] .= [0 0; 0 1]

    c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
    blocks(c⁺d)[I((1))] .= [0 0; 1 0]
    blocks(c⁺d)[I((0))] .= [0 0; -1 0]

    # cu = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
    # blocks(cu)[I((1))] .= [1 0; 0 0]
    # blocks(cu)[I((0))] .= [0 0; 0 -1] # [0 0; 0 1]

    # cd = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
    # blocks(cd)[I((1))] .= [0 1; 0 0] 
    # blocks(cd)[I((0))] .= [0 1; 0 0] # [0 -1; 0 0]

    # cu = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
    # blocks(cu)[I((1))] .= [1 0; 0 0]
    # blocks(cu)[I((0))] .= [0 0; 0 1] # [0 0; 0 1]

    # cd = TensorMap(zeros, ComplexF64, Vodd ⊗ Ps ← Ps)
    # blocks(cd)[I((1))] .= [0 1; 0 0] 
    # blocks(cd)[I((0))] .= [0 -1; 0 0] # [0 -1; 0 0]

    cu = permute(c⁺u', ((2,1), (3,)))
    cd = permute(c⁺d', ((2,1), (3,)))

    @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
    @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
    twosite = twosite_up + twosite_down

    # @tensor nup[-1; -2] := cu[2 1; -2] * c⁺u[-1; 1 2]
    # @tensor ndown[-1; -2] := cd[2 1; -2] * c⁺d[-1; 1 2]
    return twosite
end

function Hopping(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        Vup = Vect[I]((1, Q, 1) => 1)
        Vdown = Vect[I]((1, Q, -1) => 1)
    
        c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vup)
        blocks(c⁺u)[I((1, Q-P, 1))] .= 1
        blocks(c⁺u)[I((0, 2*Q-P, 0))] .= 1
        cu = TensorMap(zeros, ComplexF64, Vup ⊗ Ps ← Ps)
        blocks(cu)[I((1, Q-P, 1))] .= 1
        blocks(cu)[I((0, 2*Q-P, 0))] .= 1
        
        c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vdown)
        blocks(c⁺d)[I((1, Q-P, -1))] .= 1
        blocks(c⁺d)[I((0, 2*Q-P, 0))] .= 1
        cd = TensorMap(zeros, ComplexF64, Vdown ⊗ Ps ← Ps)
        blocks(cd)[I((1, Q-P, -1))] .= 1
        blocks(cd)[I((0, 2*Q-P, 0))] .= 1
    
        @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
        @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
        twosite = twosite_up + twosite_down

        @tensor nup[-1; -2] := cu[2 1; -2] * c⁺u[-1; 1 2]
        @tensor ndown[-1; -2] := cd[2 1; -2] * c⁺d[-1; 1 2]
        n = nup + ndown
    else
        Vs = Vect[I]((1, Q, 1 // 2) => 1)

        c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
        blocks(c⁺)[I((1, Q-P, 1 // 2))] .= 1
        blocks(c⁺)[I((0, 2*Q-P, 0))] .= sqrt(2)

        c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
        blocks(c)[I((1, Q-P, 1 // 2))] .= 1
        blocks(c)[I((0, 2*Q-P, 0))] .= sqrt(2)

        @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]

        @tensor n[-1; -2] := c[2 1; -2] * c⁺[-1; 1 2]
    end
    return twosite, n
end

function ASym_OSInteraction()
    I, Ps = ASymSpace()
    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0))] .= [0 0; 0 1]
    return onesite
end

function OSInteraction(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 2*Q-P, 0))] .= 1
    else
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 2*Q-P, 0))] .= 1
    end

    return onesite
end

# @tensor c⁺c⁻[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
# @tensor c⁺c⁻bis[-1 -2; -3 -4] := c⁺[-1; -3 1] * c⁺'[-2 1; -4]
# @show norm(c⁺c⁻bis-c⁺c⁻)