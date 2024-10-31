function rotate_psi_l90(psi)
    psi_new = copy(psi)
    psi_new[1,1] = rotl90(psi[1,2])
    psi_new[1,2] = rotl90(psi[2,2])
    psi_new[2,2] = rotl90(psi[2,1])
    psi_new[2,1] = rotl90(psi[1,1])
    return psi_new
end

function rotate_lambdas_l90(lambdas; permutation = "forward")
    if permutation == "leave_invariant"
        return [lambdas[3],
                lambdas[4],
                lambdas[5], 
                lambdas[6],
                lambdas[7],
                lambdas[8],
                lambdas[1], 
                lambdas[2]
        ]
    elseif permutation == "forward"
        return [lambdas[3],
                permute(lambdas[4], ((2,), (1,))),
                permute(lambdas[5], ((2,), (1,))),
                lambdas[6],
                lambdas[7],
                permute(lambdas[8], ((2,), (1,))),
                permute(lambdas[1], ((2,), (1,))),
                lambdas[2]
        ]
    else
        @error "Incorrect argument in rotate_lambdas_l90"
    end
end

function absorb_lambdas(left, right, lambdas; inverse = false)
    if inverse
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 4 2 3] * inv(sqrt(lambdas[1]))[1; -2] * inv(sqrt(lambdas[8]))[-4; 2] * inv(sqrt(lambdas[3]))[-5; 3] * sqrt(lambdas[2])[4; -3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 4] * inv(sqrt(lambdas[5]))[1; -2] * inv(sqrt(lambdas[3]))[2; -3] * inv(sqrt(lambdas[4]))[-4; 3] * sqrt(lambdas[2])[-5; 4]
    else
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 4 2 3] * sqrt(lambdas[1])[1; -2] * sqrt(lambdas[8])[-4; 2] * sqrt(lambdas[3])[-5; 3] * inv(sqrt(lambdas[2]))[4; -3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 4] * sqrt(lambdas[5])[1; -2] * sqrt(lambdas[3])[2; -3] * sqrt(lambdas[4])[-4; 3] * inv(sqrt(lambdas[2]))[-5; 4]
    end
    return (left_t, right_t)
end

function gauge_fix_north(psi, lambdas, base_space)
    @tensor ML[-1; -2] := psi[1,1][7; 1 11 3 5] * conj(psi[1,1][7; 8 12 9 10]) * 
    sqrt(lambdas[1])[1; 2] * sqrt(lambdas[8])[4; 3] * sqrt(lambdas[3])[6; 5] *
    conj(sqrt(lambdas[1])[8; 2]) * conj(sqrt(lambdas[8])[4; 9]) * conj(sqrt(lambdas[3])[6; 10]) * 
    sqrt(inv(lambdas[2]))[11; -2] * conj(sqrt(inv(lambdas[2]))[12; -1])

    @tensor MR[-1; -2] := psi[1,2][7; 1 3 5 11] * conj(psi[1,2][7; 8 9 10 12]) * 
    lambdas[5][1; 2] * lambdas[3][3; 4] * lambdas[4][6; 5] * 
    conj(lambdas[5][8; 2]) * conj(lambdas[3][9; 4]) * conj(lambdas[4][6; 10]) * 
    sqrt(inv(lambdas[2]))[-1; 11] * conj(sqrt(inv(lambdas[2]))[-2; 12])

    # println("check: ML and MR should be diagonal in the beginning:")
    # println("ML = $(ML)")
    # println("MR = $(MR)")

    current_space = psi[1,1].dom[2]
    I₁ = isometry(base_space, current_space)
    I₂ = isometry(current_space, base_space)

    uL, dL, uLconj = tsvd(ML) # uLconj has the correct index order
    uRconj, dR, uR = tsvd(MR) # uRconj has the correct index order
    
    @assert (norm(uL * dL * uLconj - ML) < 1e-10)
    @assert (norm(uRconj * dR * uR - MR) < 1e-10)

    @tensor uL[-1; -2] := uL[-1; 1] * I₁[1; -2]
    @tensor dL[-1; -2] := I₂[-1; 1] * dL[1; 2] * I₁[2; -2]
    @tensor uLconj[-1; -2] := I₂[-1; 1] * uLconj[1; -2]

    @tensor uRconj[-1; -2] := uRconj[-1; 1] * I₁[1; -2]
    @tensor dR[-1; -2] := I₂[-1; 1] * dR[1; 2] * I₁[2; -2]
    @tensor uR[-1; -2] := I₂[-1; 1] * uR[1; -2]

    if (norm(uLconj' - uL) > 1e-10) || (norm(uRconj' - uR) > 1e-10)
        @warn "Diagonalization failed"
    end
    @assert norm(inv(uL)-uLconj) < 1e-10
    @assert norm(inv(uR)-uRconj) < 1e-10

    λ′ = sqrt(dL) * uLconj * lambdas[2] * uRconj' * sqrt(dR)
    wL, λTilde, wR = tsvd(λ′)
    
    @tensor wL[-1; -2] := wL[-1; 1] * I₁[1; -2]
    @tensor λTilde[-1; -2] := I₂[-1; 1] * λTilde[1; 2] * I₁[2; -2]
    @tensor wR[-1; -2] := I₂[-1; 1] * wR[1; -2]

    @assert norm(inv(wR) - wR') < 1e-10
    @assert norm(inv(wL) - wL') < 1e-10

    x = wL' * sqrt(dL) * uLconj
    y = uRconj * sqrt(dR) * wR

    # println(wR * wR') # should be I    
    @assert norm(x' * x - ML) < 1e-10
    @assert norm(y * y' - MR) < 1e-10

    # lambda_new = x * lambdas[2] * y
    # # @assert (norm(lambda_new - λTilde) < 1e-5)
    # lambda_new = lambda_new / norm(lambda_new)
    lambda_new = λTilde / norm(λTilde)
    lambdas[2] = copy(lambda_new)
    @tensor psi11_new[-1; -2 -3 -4 -5] := psi[1,1][-1; -2 1 -4 -5] * inv(x)[1; 2] * sqrt(lambda_new)[2; -3]
    @tensor psi12_new[-1; -2 -3 -4 -5] := sqrt(lambda_new)[-5; 1] * inv(y)[1; 2] * psi[1,2][-1; -2 -3 -4 2]

    normalization = sqrt((norm(psi[1,1])^2+norm(psi[1,2])^2)/(norm(psi11_new)^2+norm(psi12_new)^2))
    psi[1,1] = copy(normalization*psi11_new)
    psi[1,2] = copy(normalization*psi12_new)

    @tensor ML[-1; -2] := psi[1,1][7; 1 11 3 5] * conj(psi[1,1][7; 8 12 9 10]) * 
    sqrt(lambdas[1])[1; 2] * sqrt(lambdas[8])[4; 3] * sqrt(lambdas[3])[6; 5] *
    conj(sqrt(lambdas[1])[8; 2]) * conj(sqrt(lambdas[8])[4; 9]) * conj(sqrt(lambdas[3])[6; 10]) * 
    sqrt(inv(lambdas[2]))[11; -2] * conj(sqrt(inv(lambdas[2]))[12; -1])

    @tensor MR[-1; -2] := psi[1,2][7; 1 3 5 11] * conj(psi[1,2][7; 8 9 10 12]) * 
    lambdas[5][1; 2] * lambdas[3][3; 4] * lambdas[4][6; 5] * 
    conj(lambdas[5][8; 2]) * conj(lambdas[3][9; 4]) * conj(lambdas[4][6; 10]) * 
    sqrt(inv(lambdas[2]))[-1; 11] * conj(sqrt(inv(lambdas[2]))[-2; 12])

    # println("check: ML and MR should be diagonal at the end:")
    # println("ML = $(ML)")
    # println("MR = $(MR)")
    return (psi, lambdas)
end

function simple_update_north(psi, lambdas, dτ, D, twosite_operator, base_space; gauge_fixing = false)
    U = exp(-dτ*twosite_operator)

    if gauge_fixing
        (psi, lambdas) = gauge_fix_north(psi, lambdas, base_space)
    end

    left = psi[1,1]
    right = psi[1,2]
    joint_norm_old = sqrt(norm(left)^2 + norm(right)^2)

    (left_t, right_t) = absorb_lambdas(left, right, lambdas, inverse = false)

    # Group the legs of the tensor and perform QR, LQ decomposition
    (left_index, right_index) = (3, 5)
    
    Ql, R = leftorth(left_t, (Tuple(setdiff(2:5, left_index)), (1, left_index)), alg = QR())
    L, Qr = rightorth(right_t, ((1, right_index), Tuple(setdiff(2:5, right_index))), alg = LQ())

    Ql = permute(Ql, ((4,), (1, 2, 3)))
    R = permute(R, ((2, 3), (1,)))

    @tensor left_new[-1; -2 -3 -4 -5] := Ql[1; -2 -4 -5] * R[-1 -3; 1]
    @tensor right_new[-1; -2 -3 -4 -5] := L[-1 -5; 1] * Qr[1; -2 -3 -4]

    if norm(left_new - left_t) > 1e-10
        @warn "norm difference between reconstructed and original tensor after QR decomposition is $(norm(left_new - left_t))"
    end
    if norm(right_new - right_t) > 1e-10
        @warn "norm difference between reconstructed and original tensor after LQ decomposition is $(norm(right_new - right_t))"
    end

    @tensor Θ[-1 -2; -3 -4] := R[1 2; -1] * L[4 3; -4] * U[-2 -3; 1 4] * lambdas[2][2; 3]

    (R_new, lambda_new, L_new) = tsvd(Θ, trunc = truncdim(D))

    current_space = left.dom[2]
    I₁ = isometry(base_space, current_space)
    I₂ = isometry(current_space, base_space)
    @tensor R_new[-1 -2; -3] := R_new[-1 -2; 1] * I₁[1; -3]
    @tensor lambda_new[-1; -2] := I₂[-1; 1] * lambda_new[1; 2] * I₁[2; -2]
    @tensor L_new[-1; -2 -3] := I₂[-1; 1] * L_new[1; -2 -3]

    @tensor Plnew[-1; -2 -3 -4 -5] := Ql[1; -2 -4 -5] * R_new[1 -1; -3]
    @tensor Prnew[-1; -2 -3 -4 -5] := L_new[-5; -1 1] * Qr[1; -2 -3 -4]

    # println("norm of new_lambda = $(norm(lambda_new))")
    lambdas[2] = lambda_new / norm(lambda_new)

    (left_new, right_new) = absorb_lambdas(Plnew, Prnew, lambdas, inverse = true)

    joint_norm_new = sqrt(norm(left_new)^2 + norm(right_new)^2)
    factor = joint_norm_old / joint_norm_new
    # println("factor for norm is $(joint_norm_old)/$(joint_norm_new) = $(factor)")

    psi[1,1] = left_new * factor 
    psi[1,2] = right_new * factor

    return (psi, lambdas)
end

function translate_psi_hor(psi)
    psi_new = copy(psi)
    psi_new[1,1] = psi[1,2]
    psi_new[1,2] = psi[1,1]
    psi_new[2,1] = psi[2,2]
    psi_new[2,2] = psi[2,1]
    return psi_new
end

function translate_psi_diag(psi)
    psi_new = copy(psi)
    psi_new[1,1] = psi[2,2]
    psi_new[1,2] = psi[2,1]
    psi_new[2,1] = psi[1,2]
    psi_new[2,2] = psi[1,1]
    return psi_new
end

function translate_lambdas_hor(lambdas)
    return [lambdas[5],
        lambdas[3],
        lambdas[2],
        lambdas[8],
        lambdas[1],
        lambdas[7],
        lambdas[6],
        lambdas[4]
    ]
end

function translate_lambdas_diag(lambdas)
    return [lambdas[4],
        lambdas[7],
        lambdas[6],
        lambdas[1],
        lambdas[8],
        lambdas[3],
        lambdas[2],
        lambdas[5]
    ]
end

function get_energy_CTMRG(psi, H, ctm_alg, vspace_env)
    env0 = CTMRGEnv(psi, vspace_env);
    env = leading_boundary(env0, psi, ctm_alg);
    return expectation_value(psi, H, env)
end

function get_energy_bond(left, right, lambdas, twosite)
    return PEPSKit.@autoopt @tensor left[dAt; DAtN DAtE DAtS DAtW] * conj(left[dAb; DAbN DAbE DAbS DAbW]) * 
    right[dBt; DBtN DBtE DBtS DAtE] * conj(right[dBb; DBbN DBbE DBbS DAbE]) * 
    sqrt(lambdas[1])[DAtN; D1] * conj(sqrt(lambdas[1])[DAbN; D1]) *
    sqrt(lambdas[5])[DBtN; D5] * conj(sqrt(lambdas[5])[DBbN; D5]) * 
    sqrt(lambdas[8])[D8; DAtS] * conj(sqrt(lambdas[8])[D8; DAbS]) * 
    sqrt(lambdas[4])[D4; DBtS] * conj(sqrt(lambdas[4])[D4; DBbS]) * 
    sqrt(lambdas[3])[D3L; DAtW] * conj(sqrt(lambdas[3])[D3L; DAbW]) * 
    sqrt(lambdas[3])[DBtE; D3R] * conj(sqrt(lambdas[3])[DBbE; D3R]) * twosite[dAb dBb; dAt dBt]
end

function simple_update(psi, twosite_operator, dτ, D, max_iterations, ctm_alg; χenv = 3*D, translate = false, gauge_fixing = false, printing_freq = 100, dτ_decrease = 10^(1/1000))
    base_space = psi[1,1].dom[2]
    lambdas = fill(id(base_space),8)
    energies = []
    for i = 1:max_iterations
        if (i % printing_freq) == 0 && (i != 0)
            println("Started with iteration $(i) - current energy is $(energies[end])")
        end
        if (i % 50) == 0
            energy = 0
            for i = 1:4
                (psi, lambdas) = simple_update_north(psi, lambdas, dτ, D, twosite_operator, base_space; gauge_fixing = gauge_fixing)
                energy_bond = get_energy_bond(psi[1,1], psi[1,2], lambdas, twosite_operator)
                psi = rotate_psi_l90(psi)
                lambdas = rotate_lambdas_l90(lambdas)
                energy += energy_bond
            end
            psi = translate_psi_diag(psi)
            lambdas = translate_lambdas_diag(lambdas)
            for i = 1:4
                (psi, lambdas) = simple_update_north(psi, lambdas, dτ, D, twosite_operator, base_space; gauge_fixing = gauge_fixing)
                energy_bond = get_energy_bond(psi[1,1], psi[1,2], lambdas, twosite_operator)
                psi = rotate_psi_l90(psi)
                lambdas = rotate_lambdas_l90(lambdas)
                energy += energy_bond
            end
            push!(energies, energy)
        else
            for i = 1:4
                (psi, lambdas) = simple_update_north(psi, lambdas, dτ, D, twosite_operator, base_space; gauge_fixing = gauge_fixing)
                psi = rotate_psi_l90(psi)
                lambdas = rotate_lambdas_l90(lambdas)
            end
            psi = translate_psi_diag(psi)
            lambdas = translate_lambdas_diag(lambdas)
            for i = 1:4
                (psi, lambdas) = simple_update_north(psi, lambdas, dτ, D, twosite_operator, base_space; gauge_fixing = gauge_fixing)
                psi = rotate_psi_l90(psi)
                lambdas = rotate_lambdas_l90(lambdas)
            end
        end
        dτ /= dτ_decrease
        if dτ < 1e-7
            dτ_decrease = 1
        end
    end
    return (psi, lambdas, energies)
end

function do_CTMRG(psi, H, ctm_alg, χenv)
    opt_alg = PEPSOptimize(;
        boundary_alg=ctm_alg,
        optimizer=LBFGS(4; maxiter=10, gradtol=1e-3, verbosity=2),
        gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
        reuse_env=true,)
        
    env0 = CTMRGEnv(psi, ComplexSpace(χenv));
    env_init = leading_boundary(env0, psi, ctm_alg);
    println("initial norm = $(norm(psi, env_init))")
    result = fixedpoint(psi, H, opt_alg, env_init)    
    println("Final norm = $(norm(result.peps, result.env))")
    println("Energy after CTMRG is $(result.E)")
    return  result.peps, result.E_history
end