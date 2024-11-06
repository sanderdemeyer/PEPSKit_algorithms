"""
# This function implements the Fast Full Update algorithm for the structure
#     a b
#     b a, 
# using the paper https://arxiv.org/abs/1503.05345.
"""

# struct PEPS_FFU{T<:AbstractTensorMap,E<:Number} 
#     a::T{E,1,4}
#     b::T{E,1,4}
#     C::Array{T{E,1,1},4}
#     Ta::Array{T{E,3,1},4}
#     Tb::Array{T{E,3,1},4}
# end

mutable struct PEPS_FFU
    a
    b
    C
    Ta
    Tb
end

function apply_env_bottom(psi, χenv)    
    @tensor E2[-1 -2 -3; -4] := psi.C[1][-1; 1] * psi.Tb[1][1 -2 -3; -4]
    @tensor E3[-1 -2 -3; -4] := psi.Ta[1][-1 -2 -3; 1] * psi.C[2][1; -4]

    PEPSKit.@autoopt @tensor bottom[DRN DaNa DaNb; DbNa DbNb DLN] := psi.Ta[4][DC4N DbWa DbWb; DLN] * 
    psi.b[Db; DbNa DbEa DbSa DbWa] * conj(psi.b[Db; DbNb DbEb DbSb DbWb]) * 
    psi.a[Da; DaNa DaEa DaSa DbEa] * conj(psi.a[Da; DaNb DaEb DaSb DbEb]) * 
    psi.Tb[2][DRN DaEa DaEb; DC3N] * 
    psi.C[4][DC4E; DC4N] * psi.Ta[3][DI DbSa DbSb; DC4E] * 
    psi.Tb[3][DC3W DaSa DaSb; DI] * psi.C[3][DC3N; DC3W]

    E5, Σ, E6_prel = tsvd(bottom, trunc = truncdim(χenv))
    @tensor E6[-1 -2 -3; -4] := Σ[-1; 1] * E6_prel[1; -2 -3 -4]

    # @tensor check[-1 -2 -3; -4 -5 -6] := E6[1 -4 -5; -6] * E5[-1 -2 -3; 1]
    # if (norm(check-bottom) > 1e-5)
    #     @warn "SVD of the reduced environments is not exact: $(norm(check-bottom))"
    # end 
    return [psi.Tb[4] E2 E3 psi.Ta[2] E5 E6]
end

function apply_env_left(psi, a1, b, χenv)
    joint_norm_start = sqrt(norm(psi.C[1])^2 + norm(psi.C[4])^2 + 
    norm(psi.Tb[4])^2 + norm(psi.Ta[4])^2)

    PEPSKit.@autoopt @tensor I1[DS DaSa DaSb DaEa DaEb; DE] := psi.C[1][DC1S; DC1E] * 
    psi.Tb[1][DC1E DaNa DaNb; DE] * psi.Tb[4][DS DaWa DaWb; DC1S] * 
    a1[Da; DaNa DaEa DaSa DaWa] * conj(a1[Da; DaNb DaEb DaSb DaWb])

    PEPSKit.@autoopt @tensor I3[DE; DbEa DbEb DbNa DbNb DN] := psi.C[4][DC4E; DC4N] * 
    psi.Ta[4][DC4N DbWa DbWb; DN] * psi.Ta[3][DE DbSa DbSb; DC4E] * 
    b[Db; DbNa DbEa DbSa DbWa] * conj(b[Db; DbNb DbEb DbSb DbWb])

    I2, Σ1, C1_prel = tsvd(I1, trunc = truncdim(χenv))
    C4_prel, Σ2, I4 = tsvd(I3, trunc = truncdim(χenv))
    @tensor C1new[-1; -2] := Σ1[-1; 1] * C1_prel[1; -2]
    @tensor C4new[-1; -2] := C4_prel[-1; 1] * Σ1[1; -2]

    PEPSKit.@autoopt @tensor I5[DS DI4a DI4b; DI2a DI2b DN] := I2[DL; Da Db DI2a DI2b DN] * 
    I4[DS; DI4a DI4b Da Db DL]
    # println(summary(I5))

    Tb4new, Σ3, Ta4_prel = tsvd(I5, trunc = truncdim(χenv))
    @tensor Ta4new[-1 -2 -3; -4] := Σ3[-1; 1] * Ta4_prel[1; -2 -3 -4]

    # joint_norm_end = sqrt(norm(C1new)^2 + norm(C4new)^2 + 
    # norm(Ta4new)^2 + norm(Tb4new)^2)
    # f = joint_norm_start / joint_norm_end

    psi.C[1] = C1new# * f
    psi.C[4] = C4new# * f
    psi.Tb[4] = Ta4new# * f    
    psi.Ta[4] = Tb4new# * f
    # println("Factor in apply_env_left = $(f)")

    return psi    
end

function apply_env_right(psi, a, b1, χenv)
    joint_norm_start = sqrt(norm(psi.C[2])^2 + norm(psi.C[3])^2 + 
    norm(psi.Tb[2])^2 + norm(psi.Ta[2])^2)

    PEPSKit.@autoopt @tensor I1[DW; DbWa DbWb DbSa DbSb DS] := psi.C[2][DC2W; DC2S] * 
    psi.Ta[1][DW DbNa DbNb; DC2W] * psi.Ta[2][DC2S DbEa DbEb; DS] * 
    b1[Db; DbNa DbEa DbSa DbWa] * conj(b1[Db; DbNb DbEb DbSb DbWb])

    PEPSKit.@autoopt @tensor I3[DN DaNa DaNb DaWa DaWb; DW] := psi.C[3][DC3N; DC3W] * 
    psi.Tb[2][DN DaEa DaEb; DC3N] * psi.Tb[3][DC3W DaSa DaSb; DW] * 
    a[Da; DaNa DaEa DaSa DaWa] * conj(a[Da; DaNb DaEb DaSb DaWb])
    
    C2_prel, Σ1, I2 = tsvd(I1, trunc = truncdim(χenv))
    I4, Σ2, C3_prel  = tsvd(I3, trunc = truncdim(χenv))

    @tensor C2new[-1; -2] := C2_prel[-1; 1] * Σ1[1; -2]
    @tensor C3new[-1; -2] := Σ2[-1; 1] * C3_prel[1; -2]

    PEPSKit.@autoopt @tensor I5[DN DI2a DI2b; DI4a DI4b DS] := I2[DN; DI2a DI2b Da Db DR] * 
    I4[DR Da Db DI4a DI4b; DS]

    Tb2new, Σ3, Ta2_prel = tsvd(I5, trunc = truncdim(χenv))
    @tensor Ta2new[-1 -2 -3; -4] := Σ3[-1; 1] * Ta2_prel[1; -2 -3 -4]

    # joint_norm_end = sqrt(norm(C2new)^2 + norm(C3new)^2 + 
    # norm(Ta2new)^2 + norm(Tb2new)^2)
    # f = joint_norm_start / joint_norm_end

    psi.C[2] = C2new# * f
    psi.C[3] = C3new# * f
    psi.Ta[2] = Tb2new# * f
    psi.Tb[2] = Ta2new# * f
    # println("Factor in apply_env_right = $(f)")
    return psi
end

function initialize(psi, χenv, ctm_alg)
    psi[1,1] /= norm(psi[1,1])
    psi[1,2] /= norm(psi[1,2])
    a = psi[1,1]
    b = psi[1,2]

    env0 = CTMRGEnv(psi, ComplexSpace(χenv));
    env = leading_boundary(env0, psi, ctm_alg);

    C = [env.corners[1,2,2], env.corners[2,1,2],
    env.corners[1,2,1], env.corners[2,1,1]]
    Ta = [env.edges[1,2,2], env.edges[2,1,1],
    env.edges[3,1,1], env.edges[4,2,2]]
    Tb = [env.edges[1,2,1], env.edges[2,2,1],
    env.edges[3,1,2], env.edges[4,1,2]]
    return PEPS_FFU(a, b, C, Ta, Tb)
end

function decompose_peps_hor(a, b)
    X, aR = leftorth(a, (Tuple(setdiff(2:5, 3)), (1, 3)), alg = QR())
    bL, Y = rightorth(b, ((1, 5), Tuple(setdiff(2:5, 5))), alg = LQ())
    return (X, aR, bL, Y)
end

function get_energy(envs, X, aR, B, twosite_operator)
    (E1, E2, E3, E4, E5, E6) = envs

    PEPSKit.@autoopt @tensor energy = E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DaRL] * conj(X[DE2b DE6b DE1b DLb]) * 
    B[Dpa; DE3a DE4a DE5a DaRR] * conj(B[Dpb; DE3b DE4b DE5b DRb]) *
    aR[DaRL; DaRp DaRR] * twosite_operator[Dp Dpb; DaRp Dpa] * conj(aR[DLb; Dp DRb])

    PEPSKit.@autoopt @tensor norm = E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DaRL] * conj(X[DE2b DE6b DE1b DLb]) * 
    B[Dpa; DE3a DE4a DE5a DaRR] * conj(B[Dpa; DE3b DE4b DE5b DRb]) *
    aR[DaRL; Dp DaRR] * conj(aR[DLb; Dp DRb])

    @assert abs(imag(energy/norm)) < 1e-2
    @assert abs(real(norm)-1) < 1e-1
    # println("energy from calc is $(real(energy/norm))")
    return energy/norm
end

function get_energy_and_norm(envs, A, B, twosite_operator)
    (E1, E2, E3, E4, E5, E6) = envs

    PEPSKit.@autoopt @tensor energy = E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    A[DpAa; DE2a Dma DE6a DE1a] * conj(A[DpAb; DE2b Dmb DE6b DE1b]) * 
    B[DpBa; DE3a DE4a DE5a Dma] * conj(B[DpBb; DE3b DE4b DE5b Dmb]) *
    twosite_operator[DpAb DpBb; DpAa DpBa]

    PEPSKit.@autoopt @tensor norm = E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    A[DpA; DE2a Dma DE6a DE1a] * conj(A[DpA; DE2b Dmb DE6b DE1b]) * 
    B[DpB; DE3a DE4a DE5a Dma] * conj(B[DpB; DE3b DE4b DE5b Dmb])

    return energy, norm
end

function get_reduced_update_tensors_fix_B_hor(envs, X, aR, B, B̃, U)
    (E1, E2, E3, E4, E5, E6) = envs

    PEPSKit.@autoopt @tensor R[DLb DRb; DLa DRa] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DLa] * conj(X[DE2b DE6b DE1b DLb]) * 
    B̃[Dp; DE3a DE4a DE5a DRa] * conj(B̃[Dp; DE3b DE4b DE5b DRb])

    PEPSKit.@autoopt @tensor S[DLb DRb; Dp] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DaRL] * conj(X[DE2b DE6b DE1b DLb]) * 
    B[Dpa; DE3a DE4a DE5a DaRR] * conj(B̃[Dpb; DE3b DE4b DE5b DRb]) *
    aR[DaRL; DaRp DaRR] * U[Dp Dpb; DaRp Dpa]

    return R, S
end

function get_reduced_update_tensors_fix_A_hor(envs, Y, bL, A, Ã, U)
    (E1, E2, E3, E4, E5, E6) = envs

    PEPSKit.@autoopt @tensor R[DLb DRb; DLa DRa] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    Ã[Dp; DE2a DLa DE6a DE1a] * conj(Ã[Dp; DE2b DLb DE6b DE1b]) * 
    Y[DRa; DE3a DE4a DE5a] * conj(Y[DRb; DE3b DE4b DE5b])

    PEPSKit.@autoopt @tensor S[DLb DRb; Dp] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    A[Dpa; DE2a DbLL DE6a DE1a] * conj(A[Dpb; DE2b DLb DE6b DE1b]) * 
    Y[DbLR; DE3a DE4a DE5a] * conj(Y[DRb; DE3b DE4b DE5b]) *
    bL[DbLp; DbLL DbLR] * U[Dpb Dp; Dpa DbLp]
    return R, S
end

function apply_right(A, x)
    @tensor x_new[-1 -2; -3] := A[-1 -2; 1 2] * x[1 2; -3]
    return x_new
end

function update_Ã(envs, X, aR, B, B̃, U)
    R, S = get_reduced_update_tensors_fix_B_hor(envs, X, aR, B, B̃, U)
    aR = permute(aR, ((1,3),(2,)))
    aRnew, _ = linsolve(v -> apply_right(R, v), S, aR)
    return permute(aRnew, ((1,),(3,2)))
end

function update_B̃(envs, Y, bL, A, Ã, U)
    R, S = get_reduced_update_tensors_fix_A_hor(envs, Y, bL, A, Ã, U)
    bL = permute(bL, ((2,3),(1,)))
    bLnew, _ = linsolve(v -> apply_right(R, v), S, bL)
    return permute(bLnew, ((3,1),(2,)))
end

function update_tensors(psi, twosite_operator, U, envs; max_iter_update = 100)
    (X, aR, bL, Y) = decompose_peps_hor(psi.a, psi.b)

    A = psi.a
    B = psi.b
    joint_norm_start = sqrt(norm(A)^2 + norm(B)^2)

    ãR = aR
    b̃L = bL
    Ã = A
    B̃ = B

    for i = 1:max_iter_update
        @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
        @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]

        ãRnew = update_Ã(envs, X, aR, B, B̃, U)
        b̃Lnew = update_B̃(envs, Y, bL, A, Ã, U)

        norm_dif = (norm(ãRnew - ãR), norm(b̃Lnew - b̃L))

        @tensor Aold[-1; -2 -3 -4 -5] := X[-2 -4 -5; 1] * ãR[1; -1 -3]
        @tensor Bold[-1; -2 -3 -4 -5] := b̃L[-1 -5; 1] * Y[1; -2 -3 -4]
        @tensor Anew[-1; -2 -3 -4 -5] := X[-2 -4 -5; 1] * ãRnew[1; -1 -3]
        @tensor Bnew[-1; -2 -3 -4 -5] := b̃Lnew[-1 -5; 1] * Y[1; -2 -3 -4]

        norm_dif = (norm(Anew - Aold), norm(Bnew - Bold))

        if (norm_dif[1] < 1e-10) && (norm_dif[2] < 1e-10)
            # println("Converged after $(i) iterations")
            # println("Norm differences were $(norm_dif[1]) and $(norm_dif[2])")
            energy = get_energy(envs, X, aR, B, twosite_operator)

            # joint_norm_end = sqrt(norm(A)^2 + norm(B)^2)
            # f = joint_norm_start / (joint_norm_end)^(1/4)
            # println("factor = $(f)")
            return (A, B) #(A*f, B*f)
        else
            ãR = ãRnew
            b̃L = b̃Lnew
            A = Ã
            B = B̃
        end
        if (i == max_iterations)
            @warn("Not converged after $(max_iter_update) iterations. Norm differences are $(norm_dif[1]) and $(norm_dif[2])")
            @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
            @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]
        end
    end

    # joint_norm_end = sqrt(norm(A)^2 + norm(B)^2)
    # f = joint_norm_start / (joint_norm_end)^(1/4)
    # println("factor = $(f)")
    return (A, B) #(A*f, B*f)
end

function update_bond(psi, twosite_operator, U, χenv)
    envs = apply_env_bottom(psi, χenv)
    energy_bond, _ = get_energy_and_norm(envs, psi.a, psi.b, twosite_operator)

    a1, b1 = update_tensors(psi, twosite_operator, U, envs)
    psi = apply_env_left(psi, a1, psi.b, χenv)
    psi = apply_env_right(psi, psi.a, b1, χenv)

    psi.a = b1
    psi.b = a1

    psi = normalize_psi(psi)
    psi = normalize_env(psi, twosite_operator, χenv)
    return psi, energy_bond
end

function rotl90_state(psi)
    psi.C = circshift(psi.C, -1)
    (psi.Ta, psi.Tb) = (circshift(psi.Tb, -1), circshift(psi.Ta, -1))
    (psi.a, psi.b) = (rotl90(psi.b), rotl90(psi.a))
    return psi
end

function FFU_step(psi, twosite_operator, U, χenv)
    psi, energy_bondr = update_bond(psi, twosite_operator, U, χenv)
    psi, energy_bondl = update_bond(psi, twosite_operator, U, χenv)
    psi = rotl90_state(psi)
    psi, energy_bondu = update_bond(psi, twosite_operator, U, χenv)
    psi, energy_bondd = update_bond(psi, twosite_operator, U, χenv)
    return psi, (energy_bondr + energy_bondl + energy_bondu + energy_bondd)
end

function normalize_env(psi, twosite_operator, χenv)
    envs = apply_env_bottom(psi, χenv)
    energy, norm = get_energy_and_norm(envs, psi.a, psi.b, twosite_operator)
    # println("norm = $(norm)")
    psi.C /= norm^(1/12)
    psi.Ta /= norm^(1/12)
    psi.Tb /= norm^(1/12)
    return psi
end

function normalize_psi(psi)
    joint_norm = sqrt(2*norm(psi.a)^2 + 2*norm(psi.b)^2)
    psi.a /= joint_norm
    psi.b /= joint_norm
    return psi
end

function FFU(psi, twosite_operator, dτ, max_iterations, ctm_alg; χenv = 3*D^2, gauge_fixing = false)
    U = exp(-dτ*twosite_operator)
    if gauge_fixing
        println("Gauge fixing not yet implemented")
    end

    psi = initialize(psi, χenv, ctm_alg)
    println("Start with normalization")
    psi = normalize_env(psi, twosite_operator, χenv)

    for i = 1:max_iterations
        println("Started iteration $(i)")
        psi, energy = FFU_step(psi, twosite_operator, U, χenv)
        println("After iteration $(i), energy = $(energy)")
    end
    return psi
end

