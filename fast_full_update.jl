include("utility.jl")

dτ = 0.1
χ = 2
D = 2
χenv = 4

δ = dτ

Js = (-1, 1, -1)

unitcell = (2, 2)
H = square_lattice_heisenberg(; Jx = Js[1], Jy = Js[2], Jz = Js[3], unitcell = (2,2))

psi = InfinitePEPS(2, D; unitcell)
psi = (1/norm(psi))*psi

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:sequential,
)

env0 = CTMRGEnv(psi, ComplexSpace(χenv));
env = leading_boundary(env0, psi, ctm_alg);

function get_reduced_env(psi, env, χenv)
    E1 = env.edges[4,1,2]
    E4 = env.edges[2,1,1]

    @tensor E2[-1 -2 -3; -4] := env.edges[1,2,1][1 -2 -3; -4] * env.corners[1,2,2][-1; 1]
    @tensor E3[-1 -2 -3; -4] := env.edges[1,2,2][-1 -2 -3; 1] * env.corners[2,2,1][1; -4]

    (E5, E6) = apply_environment_south(psi, env, χenv)

    return (E1, E2, E3, E4, E5, E6)
end

function apply_env_south_ctmrg(psi, env, alg)
    psi, env = apply_ctmrg(psi, env, alg)

    # C4 = env.corners[4,1,1]
    # C3 = env.corners[3,1,2]
    # TD3 = env.edges[3,1,1]
    # TC3 = env.edges[3,1,2]

    # C4 = env.corners[4,2,2]
    # C3 = env.corners[3,2,1]
    # TD3 = env.edges[3,1,2]
    # TC3 = env.edges[3,2,2]

    # C4 = env.corners[4,1,2]
    # C3 = env.corners[3,1,1]
    # TD3 = env.edges[3,2,2]
    # TC3 = env.edges[3,1,2]

    C4 = env.corners[4,1,2]
    C3 = env.corners[3,1,1]
    TD3 = env.edges[3,1,1]
    TC3 = env.edges[3,1,2]

    @tensor E5[-1 -2 -3; -4] := TC3[1 -2 -3; -4] * C3[-1; 1]
    @tensor E6[-1 -2 -3; -4] := C4[1; -4] * TD3[-1 -2 -3; 1]
    return (E5, E6)
end

function apply_environment_south(psi, env, χenv)
    PEPSKit.@autoopt @tensor south[χW DDb DDa; DCa DCb χE] := psi[2,1][p1; DDa D1 D2 D3] * psi[2,2][p2; DCa D7 D8 D1] * 
    conj(psi[2,1][p1; DDb D4 D5 D6]) * conj(psi[2,2][p2; DCb D9 D10 D4]) * 
    env.edges[4,2,2][χSW; D3 D6 χW] * env.corners[4,2,1][χ1; χSW] * 
    env.edges[3,1,1][χ2; D2 D5 χ1] * env.edges[3,1,2][χ3; D8 D10 χ2] *
    env.edges[2,2,1][χE; D7 D9 χSE] * env.corners[3,1,1][χSE; χ3]
    
    U, S, E5 = tsvd(south)
    E5 = permute(E5, ((4,2,3), (1,)))
    @tensor E6[-1 -2 -3; -4] := U[-4 -3 -2; 1] * S[1; -1]
    # @tensor E4[-4; -3 -2 -1] := U[-4 -2 -3; 1] * S[1; -1]
    return E5, E6
end

function get_reduced_update_tensors_fix_B_hor(envs, X, aR, B, B̃, U)
    (E1, E2, E3, E4, E5, E6) = envs

    # PEPSKit.@autoopt @tensor R[DLa DLb; DRa DRb] := E1[χSW DE1a DE1b; χNW] *
    PEPSKit.@autoopt @tensor R[DLb DRb; DLa DRa] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DLa] * conj(X[DE2b DE6b DE1b DLb]) * 
    B̃[Dp; DE3a DE4a DE5a DRa] * conj(B̃[Dp; DE3b DE4b DE5b DRb])
    # X[DLa; DE2a DE6a DE1a] * conj(X[DLb; DE2b DE6b DE1b]) * 

    # PEPSKit.@autoopt @tensor S[DLb; DRb Dp] := E1[χSW DE1a DE1b; χNW] *
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
    # Ã[Dp; DE2a DE6a DE1a DLa] * conj(Ã[Dp; DE2b DE6b DE1b DLb]) * 
    # Y[DE3a DE4a DE5a; DRa] * conj(Y[DE3b DE4b DE5b; DRb])

    PEPSKit.@autoopt @tensor S[DLb DRb; Dp] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    A[Dpa; DE2a DbLL DE6a DE1a] * conj(A[Dpb; DE2b DLb DE6b DE1b]) * 
    Y[DbLR; DE3a DE4a DE5a] * conj(Y[DRb; DE3b DE4b DE5b]) *
    bL[DbLp; DbLL DbLR] * U[Dpb Dp; Dpa DbLp]
    # A[Dpa; DE2a DE6a DE1a DaRL] * conj(A[Dpb; DE2b DE6b DE1b DLb]) * 
    # Y[DE3a DE4a DE5a; DaRR] * conj(Y[DE3b DE4b DE5b; DRb]) *
    # bL[DaRL; DbLp DaRR] * U[Dpb Dp; Dpa DbLp]
    return R, S
end

function reduced_update_OLD(psi, env, δ)
    (left_index, right_index) = (3, 5)

    U = get_gate(δ, ((-1, 1, -1)))

    X, aR = leftorth(psi[1,1], (Tuple(setdiff(2:5, left_index)), (1, left_index)), alg = QR())
    bL, Y = rightorth(psi[1,2], ((1, right_index), Tuple(setdiff(2:5, right_index))), alg = LQ())
    envs = get_reduced_env(psi, env, χenv)

    B̃ = psi[1,2]
    R, S = get_reduced_update_tensors_fix_B(envs, X, aR, psi[1,2], B̃, U)

    println(summary(aR))
    aR = permute(aR, ((1,3),(2,)))
    aRnew, _ = linsolve(v -> apply_right(R, v), S, aR)
    aRnew2 = permute(aRnew, ((1,),(3,2)))
    return aRnew2
end

function decompose_peps_hor(psi)
    X, aR = leftorth(psi[1,1], (Tuple(setdiff(2:5, 3)), (1, 3)), alg = QR())
    bL, Y = rightorth(psi[1,2], ((1, 5), Tuple(setdiff(2:5, 5))), alg = LQ())
    return (X, aR, bL, Y)
end

function apply_right(A, x)
    @tensor x_new[-1 -2; -3] := A[-1 -2; 1 2] * x[1 2; -3]
    return x_new
end

function update_Ã(envs, X, aR, B, B̃, U)
    R, S = get_reduced_update_tensors_fix_B(envs, X, aR, B, B̃, U)
    aR = permute(aR, ((1,3),(2,)))
    aRnew, _ = linsolve(v -> apply_right(R, v), S, aR)
    return permute(aRnew, ((1,),(3,2)))
end

function update_B̃(envs, Y, bL, A, Ã, U)
    R, S = get_reduced_update_tensors_fix_A(envs, Y, bL, A, Ã, U)
    bL = permute(bL, ((2,3),(1,)))
    bLnew, _ = linsolve(v -> apply_right(R, v), S, bL)
    return permute(bLnew, ((3,1),(2,)))
end

function update_alternating(psi, env, χenv, δ; max_iterations = 100)
    # check how the distance decreases?
    envs = get_reduced_env(psi, env, χenv)

    U = get_gate(δ, ((-1, 1, -1)))

    (X, aR, bL, Y) = decompose_peps(psi)

    A = psi[1,1]
    B = psi[1,2]
    ãR = aR
    b̃L = bL
    Ã = A
    B̃ = B

    for i = 1:max_iterations
        println("i = $(i)")
        @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
        @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]

        ãRnew = update_Ã(envs, X, aR, B, B̃, U)

        b̃Lnew = update_B̃(envs, Y, bL, A, Ã, U)

        if (norm(ãRnew - ãR) < 1e-5) && (norm(b̃Lnew - b̃L) < 1e-5)
            @tensor psi[1,1][-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
            @tensor psi[1,2][-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]
            return psi
        else
            println("norm difference for aR = $(norm(ãRnew - ãR)). Norm difference for bL = $(norm(b̃Lnew - b̃L))")
            ãR = ãRnew
            b̃L = b̃Lnew
        end
    end
    @warn("Not converged after $(max_iterations) iterations.")
    @tensor psi[1,1][-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
    @tensor psi[1,2][-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]
    return psi
end

function apply_ctmrg(state, envs, alg)
    enlarged_envs = ctmrg_expand(state, envs, alg)
    projectors, _ = ctmrg_projectors(enlarged_envs, envs, alg)
    envs = ctmrg_renormalize(projectors, state, envs, alg)
    return state, envs
end

function update_link(psi, env, χenv, δ, alg)
    psi = update_alternating(psi, env, χenv, δ)

    psi_old = copy(psi)
    env_corners_old = copy(env.corners)
    env_edges_old = copy(env.edges)
    psi, env = apply_ctmrg(psi, env, alg)

    c = reshape(norm.(env_corners_old-env.corners), 4, 2, 2)
    e = reshape(norm.(env_edges_old-env.edges), 4, 2, 2)

    for i = 1:4
        println("i = $(i)")
        println(c[i,:,:])
        println(e[i,:,:])
    end
    return 0
end

update_link(psi, env, χenv, δ, ctm_alg)

# reduced_update(psi[1,1], psi[1,2], env)

# (left_index, right_index) = (3, 5)
# test = TensorMap(randn, ℂ^2, ℂ^3 ⊗ ℂ^4 ⊗ (ℂ^5)' ⊗ (ℂ^6)')
# X, aR = leftorth(test, (Tuple(setdiff(2:5, left_index)), (1, left_index)), alg = QR())
# bL, Y = rightorth(test, ((1, right_index), Tuple(setdiff(2:5, right_index))), alg = LQ())

# psi2 = copy(psi)

# (E5a, E6a) = apply_env_south_ctmrg(psi, env, ctm_alg)

# (E5b, E6b) = apply_environment_south(psi2, env, χenv)

# @tensor southa[-1 -2 -3 -4 -5; -6] := E5a[-1 -2 -3; 1] * E6a[1 -4 -5; -6]
# @tensor southb[-1 -2 -3 -4 -5; -6] := E5b[-1 -2 -3; 1] * E6b[1 -4 -5; -6]

# # permute(southa, ((1, 2, 3, 4, 5), (6,)))
# # permute(southa, ((1, 4, 3, 2, 5), (6,)))
# # permute(southa, ((1, 2, 5, 4, 3), (6,)))
# permute(southa, ((1, 4, 5, 2, 3), (6,)))

# println(norm((1/norm(southa)*southa-(1/norm(southb))*southb)))

# println("Done")