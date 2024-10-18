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

function get_energy(psi_base, H, ctm_alg, χenv)
    env0 = CTMRGEnv(psi_base, ComplexSpace(χenv));
    env = leading_boundary(env0, psi_base, ctm_alg);
    return expectation_value(psi, H, env)
end

function apply_right(A, x)
    @tensor x_new[-1 -2; -3] := A[-1 -2; 1 2] * x[1 2; -3]
    return x_new
end

function apply_ctmrg(state, env, alg)
    enlarged_envs = ctmrg_expand(state, env, alg)
    projectors, _ = ctmrg_projectors(enlarged_envs, env, alg)
    env = ctmrg_renormalize(projectors, state, env, alg)
    return env
end

function apply_ctmrg_and_rotate(state, env, alg)
    state = rotl90(state)
    env = rotl90(env)
    env = apply_ctmrg(state, env, alg)

    state = rotl90(state)
    state = rotl90(state)
    env = rotl90(env)
    env = rotl90(env)

    env = apply_ctmrg(state, env, alg)

    state = rotl90(state)
    env = rotl90(env)
    return env
end

function translate(state, env)
    (state[1,:], state[2,:]) = (state[2,:], state[1,:])
    (env.edges[:,1,:], env.edges[:,2,:]) = (env.edges[:,2,:], env.edges[:,1,:])
    (env.corners[:,1,:], env.corners[:,2,:]) = (env.corners[:,2,:], env.corners[:,1,:])
    return state, env
end

function get_env_tensors(env, c)
    # (E2, E3, E5, E6) = (envs.edges[2,1,mod1(c+1,2)], envs.edges[2,2,mod1(c+1,2)], envs.edges[2,1,mod1(c-1,2)], envs.edges[2,2,mod1(c-1,2)])
    (E2, E3, E5, E6) = map(v -> env.edges[2,v[1],mod1(c+v[2],2)], [(1,1), (2,1), (1,-1), (2,-1)])
    @tensor E1[-1 -2 -3; -4] :=  env.corners[1,2,mod1(c-1,2)][-1; 1] * env.edges[1,2,c][1 -2 -3; 2] * env.corners[2,2,mod1(c+1,2)][2; -4]
    @tensor E4[-1 -2 -3; -4] :=  env.corners[1,1,mod1(c-1,2)][-1; 1] * env.edges[1,1,c][1 -2 -3; 2] * env.corners[2,1,mod1(c+1,2)][2; -4]
    
    return (E1, E2, E3, E4, E5, E6)
end

function get_reduced_update_tensors_fix_B_ver(envs, X, aR, B, B̃, U)
    (E1, E2, E3, E4, E5, E6) = envs
    
    PEPSKit.@autoopt @tensor R[DNb DSb; DNa DSa] := E1[χNW DE1a DE1b χNE] * 
    E2[χNE DE2a DE2b; χE] * E3[χE DE3a DE3b; χSE] * E4[χSE DE4b DE4a; χSW] * 
    E5[χSW DE5b DE5a; χW] * E6[χW DE6b DE6a; χNW] * 
    X[DE1a DE2a DE6a; DNa] * conj(X[DE1b DE2b DE6b; DNb]) * 
    B̃[Dp; DSa DE3a DE4a DE5a] * conj(B̃[Dp; DSb DE3b DE4b DE5b])

    PEPSKit.@autoopt @tensor S[DNb DSb; Dp] := E1[χNW DE1a DE1b; χNE] * 
    E2[χNE DE2a DE2b; χE] * E3[χE DE3a DE3b; χSE] * E4[χSE DE4b DE4a; χSW] * 
    E5[χSW DE5b DE5a; χW] * E6[χW DE6b DE6a; χNW] * 
    X[DE1a DE2a DE6a; DNa] * conj(X[DE1b DE2b DE6b; DNb]) * 
    B[Dpa; DSa DE3a DE4a DE5a] * conj(B̃[Dpb; DSb DE3b DE4b DE5b]) * 
    aR[DNa; DaRp DSa] * U[Dp Dpb; DaRp Dpa]
    
    return R, S
end

function get_reduced_update_tensors_fix_A_ver(envs, Y, bL, A, Ã, U)
    (E1, E2, E3, E4, E5, E6) = envs
    
    PEPSKit.@autoopt @tensor R[DNb DSb; DNa DSa] := E1[χNW DE1a DE1b χNE] * 
    E2[χNE DE2a DE2b; χE] * E3[χE DE3a DE3b; χSE] * E4[χSE DE4b DE4a; χSW] * 
    E5[χSW DE5b DE5a; χW] * E6[χW DE6b DE6a; χNW] * 
    Ã[Dp; DE1a DE2a DNa DE6a] * conj(Ã[Dp; DE1b DE2b DNb DE6b]) * 
    Y[DSa; DE3a DE4a DE5a] * conj(Y[DSb; DE3b DE4b DE5b])
    # Y[DSa DE3a DE4a; DE5a] * conj(Y[DSb DE3b DE4b; DE5b])

    PEPSKit.@autoopt @tensor S[DNb DSb; Dp] := E1[χNW DE1a DE1b; χNE] * 
    E2[χNE DE2a DE2b; χE] * E3[χE DE3a DE3b; χSE] * E4[χSE DE4b DE4a; χSW] * 
    E5[χSW DE5b DE5a; χW] * E6[χW DE6b DE6a; χNW] * 
    A[Dpa; DE1a DE2a DNa DE6a] * conj(Ã[Dpb; DE1b DE2b DNb DE6b]) * 
    Y[DSa; DE3a DE4a DE5a] * conj(Y[DSb; DE3b DE4b DE5b]) *
    bL[DbLp DNa; DSa] * U[Dpb Dp; Dpa DbLp]
     #  U[Dp Dpb; DaRp Dpa]

    # X[DE1a DE2a DE6a; DNa] * conj(X[DE1b DE2b DE6b; DNb]) * 
    # B[Dpa; DSa DE3a DE4a DE5a] * conj(B̃[Dpb; DSb DE3b DE4b DE5b]) * 
    # aR[DNa; DaRp DSa] * U[Dp Dpb; DaRp Dpa]
    
    return R, S
end


function get_reduced_update_tensors_fix_hor(envs, X, aR, B, B̃, U)
    (E1, E2, E3, E4, E5, E6) = envs

    # PEPSKit.@autoopt @tensor R[DLa DLb; DRa DRb] := E1[χSW DE1a DE1b; χNW] *
    PEPSKit.@autoopt @tensor R[DLb DRb; DLa DRa] := E1[χSW DE1a DE1b; χNW] *
    E2[χNW DE2a DE2b; χN] * E3[χN DE3a DE3b; χNE] * E4[χNE DE4a DE4b; χSE] * 
    E5[χSE DE5a DE5b; χS] * E6[χS DE6a DE6b; χSW] * 
    X[DE2a DE6a DE1a; DLa] * conj(X[DE2b DE6b DE1b DLb]) * 
    B̃[Dp; DRa DE3a DE4a DE5a] * conj(B̃[Dp; DRb DE3b DE4b DE5b])
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

function get_reduced_update_tensors_fix_hor(envs, Y, bL, A, Ã, U)
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

function update_Ã(envs, X, aR, B, B̃, U)
    R, S = get_reduced_update_tensors_fix_B_ver(envs, X, aR, B, B̃, U)
    aR = permute(aR, ((1,3),(2,)))
    aRnew, _ = linsolve(v -> apply_right(R, v), S, aR)
    return permute(aRnew, ((1,),(3,2)))
end

function update_B̃(envs, Y, bL, A, Ã, U)
    R, S = get_reduced_update_tensors_fix_A_ver(envs, Y, bL, A, Ã, U)
    bL = permute(bL, ((2,3),(1,)))
    bLnew, _ = linsolve(v -> apply_right(R, v), S, bL)
    return permute(bLnew, ((3,1),(2,)))
end

function decompose_peps_ver(psi)
    # testt = TensorMap(randn, ℂ^2, ℂ^3 ⊗ ℂ^4 ⊗ (ℂ^5)' ⊗ (ℂ^6)')
    # X, aR = leftorth(testt, (Tuple(setdiff(2:5, 4)), (1, 4)), alg = QR())
    # bL, Y = rightorth(testt, ((1, 2), Tuple(setdiff(2:5, 2))), alg = LQ())

    X, aR = leftorth(psi[1,1], (Tuple(setdiff(2:5, 4)), (1, 4)), alg = QR())
    bL, Y = rightorth(psi[1,2], ((1, 2), Tuple(setdiff(2:5, 2))), alg = LQ())
    return (X, aR, bL, Y)
end

function get_norm_tensor_ver(envs, X, Y)
    (E1, E2, E3, E4, E5, E6) = envs

    PEPSKit.PEPSKit.@autoopt @tensor Norm[DNa DSa; DNb DSb] := E1[χNW DE1a DE1b χNE] * 
    E2[χNE DE2a DE2b; χE] * E3[χE DE3a DE3b; χSE] * E4[χSE DE4b DE4a; χSW] * 
    E5[χSW DE5b DE5a; χW] * E6[χW DE6b DE6a; χNW] * 
    X[DE1a DE2a DE6a; DNa] * conj(X[DE1b DE2b DE6b; DNb]) * 
    Y[DSa; DE3a DE4a DE5a] * conj(Y[DSb; DE3b DE4b DE5b])

    Ñorm = (Norm + Norm')/2

    println("is N hermitian? $(norm(Ñorm - Ñorm'))")
    @assert (norm(Ñorm - Ñorm') < 1e-10)
    return Ñorm
end

function update_tensor_with_gauge(envs, X, Y, aR, bL)
    Ñorm = get_norm_tensor_ver(envs, X, Y)


    @assert (norm(Ñorm - Ñorm') < 1e-10)
    W, Σ, V = tsvd(Ñorm)

    @tensor Normnew[-1 -2; -3 -4] := W[-1 -2; 1] * Σ[1; 2] * conj(W[-3 -4; 2])

    

    println(norm(Normnew - Ñorm))
    println(norm(Normnew + Ñorm))
    @assert (norm(Normnew - Ñorm) < 1e-14)
    println(summary(W))
    println(summary(W'))
    println(a)

    println(summary(V))
    println(Σ)
    println(norm(W - V'))
    println(norm(W + V'))
    println(norm(W - im*V'))
    println(norm(W + im*V'))

    Z = W * sqrt(Σ)

    @assert norm(Z * Z' - Ñorm) < 1e-10

    println(summary(Z))
    println("that was Z")
    println(a)
    return 0
end

function update_alternating(psi, env, U, c; max_iterations = 1000)
    # check how the distance decreases?

    envs = get_env_tensors(env, c)

    (X, aR, bL, Y) = decompose_peps_ver(psi)

    A = psi[1,1]
    B = psi[1,2]
    ãR = aR
    b̃L = bL
    Ã = A
    B̃ = B

    println("norms are $(norm(A)), $(norm(B))")
    norm_dif = (1, 1)
    for i = 1:max_iterations
        @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
        @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
        # @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
        # @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]

        # ãRnew = update_Ã(envs, X, aR, B, B̃, U)
        # b̃Lnew = update_B̃(envs, Y, bL, A, Ã, U)

        # Norm = get_norm_tensor_ver(envs, X, Y)
        (ãRnew, b̃Lnew, X, Y) = update_tensor_with_gauge(envs, X, Y, aR, bL)

        norm_dif = (norm(ãRnew - ãR), norm(b̃Lnew - b̃L))
        if (norm_dif[1] < 1e-5) && (norm_dif[2] < 1e-5)
            # @tensor A[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
            # @tensor B[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
            println("Converged after $(i) iterations")
            return (A, B)
        else
            # println("norm difference for aR = $(norm(ãRnew - ãR)). Norm difference for bL = $(norm(b̃Lnew - b̃L))")
            ãR = ãRnew
            b̃L = b̃Lnew
        end
    end
    @warn("Not converged after $(max_iterations) iterations. Norm differences are $(norm_dif[1]) and $(norm_dif[2])")
    @tensor A[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
    @tensor B[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
    return (A, B)
end

function update_tensors(state, env, alg, U)
    env_new = apply_ctmrg(state, env, alg)

    (A, D) = update_alternating(state, env_new, U, 1)
    (B, C) = update_alternating(state, env_new, U, 2)

    state[:] = [A B; D C]

    env = apply_ctmrg_and_rotate(state, env, alg)

    return state, env
end

function FFU(state, env, alg; max_iterations = 10, χenv = 8, δ = 0.01)
    Js = (-1, 1, -1)
    U = get_gate(δ, Js)
    H = square_lattice_heisenberg(; Jx = Js[1], Jy = Js[2], Jz = Js[3], unitcell = (2,2))

    for i = 1:max_iterations
        println("FFU iteration $(i)")
        state, env = update_tensors(state, env, alg, U)
        state, env = translate(state, env)
        state, env = update_tensors(state, env, alg, U)
        (state,env) = rotl90.((state, env))
        state, env = update_tensors(state, env, alg, U)
        state, env = translate(state, env)
        state, env = update_tensors(state, env, alg, U)
        (state,env) = rotl90.(rotl90.(rotl90.((state, env))))

        energy_else = expectation_value(psi, H, env)
        println("Energy after FFU is $(energy_else), or:")
        energy = get_energy(deepcopy(state), H, alg, χenv)
        println("$(energy)")


    end
    return (state, env);
end


ctm_alg_first = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg_first,
    optimizer=LBFGS(4; maxiter=20, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)
# result = fixedpoint(psi, H, opt_alg, env)
# state, env = FFU(result.peps, result.env, ctm_alg);

state, env = FFU(psi, env, ctm_alg);
println("Done")

test = TensorMap(randn, ℂ^2 ⊗ ℂ^3, ℂ^4 ⊗ ℂ^5)
println(summary(test))
println(summary(test'))
