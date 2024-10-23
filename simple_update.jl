include("utility.jl")
using JLD2

dτ = 1e-5
D = 2
χ = D
χenv = 2

unitcell = (2, 2)
max_iterations = 3000

Js = (-1, 1, -1)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

H = heisenberg_XYZ(InfiniteSquare(unitcell...); Jx=-1, Jy=1, Jz=-1) # sublattice rotation to obtain single-site unit cell

psi = normalize(InfinitePEPS(2, D; unitcell))
# psi = (1/norm(psi))*psi # Good normalization?

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
    if inverse # Not yet changed to new convention
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 4 2 3] * inv(sqrt(lambdas[1]))[1; -2] * inv(sqrt(lambdas[8]))[-4; 2] * inv(sqrt(lambdas[3]))[-5; 3] * sqrt(lambdas[2])[4; -3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 4] * inv(sqrt(lambdas[5]))[1; -2] * inv(sqrt(lambdas[3]))[2; -3] * inv(sqrt(lambdas[4]))[-4; 3] * sqrt(lambdas[2])[-5; 4]
    else
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 4 2 3] * sqrt(lambdas[1])[1; -2] * sqrt(lambdas[8])[-4; 2] * sqrt(lambdas[3])[-5; 3] * inv(sqrt(lambdas[2]))[4; -3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 4] * sqrt(lambdas[5])[1; -2] * sqrt(lambdas[3])[2; -3] * sqrt(lambdas[4])[-4; 3] * inv(sqrt(lambdas[2]))[-5; 4]
    end
    return (left_t, right_t)
end


function simple_update_north(psi, lambdas, dτ, χ, Js, base_space)
    U = get_gate(dτ, Js)

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

    (R_new, lambda_new, L_new) = tsvd(Θ, trunc = truncdim(χ))

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

function translate_psi(psi)
    psi_new = copy(psi)
    psi_new[1,1] = psi[1,2]
    psi_new[1,2] = psi[1,1]
    psi_new[2,1] = psi[2,2]
    psi_new[2,2] = psi[2,1]
    return psi_new
end

function translate_lambdas(lambdas)
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

function get_energy(psi, H, ctm_alg, χenv)
    env0 = CTMRGEnv(psi, ComplexSpace(χenv));
    env = leading_boundary(env0, psi, ctm_alg);
    return expectation_value(psi, H, env)
end

function simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; χenv = 3*χ, translate = false)
    lambdas = fill(id(ℂ^χ),8)
    base_space = psi[1,1].dom[2]

    energies = []
    # Do gauge fix
    for i = 1:max_iterations
        for i = 1:4
            (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, Js, base_space)
            psi = rotate_psi_l90(psi)
            lambdas = rotate_lambdas_l90(lambdas)
        end
        psi = translate_psi(psi)
        lambdas = translate_lambdas(lambdas)
        for i = 1:4
            (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, Js, base_space)
            psi = rotate_psi_l90(psi)
            lambdas = rotate_lambdas_l90(lambdas)
        end
        if mod(i, 50) == 0
            energy = get_energy(deepcopy(psi), H, ctm_alg, χenv)
            println("Energy after SU step $(i) is $(energy)")
            psi = normalize(psi)
            push!(energies, energy)
        end
    end
    return (psi, lambdas, energies)
end

function do_CTMRG(psi, H, ctm_alg, χenv)

    opt_alg = PEPSOptimize(;
        boundary_alg=ctm_alg,
        optimizer=LBFGS(4; maxiter=25, gradtol=1e-3, verbosity=2),
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

mkdir("test_SU")

(psi, energies) = do_CTMRG(psi, H, ctm_alg, χenv)
file = jldopen("test_SU/1_CTMRG", "w")
file["energies"] = energies
close(file)

(psi, lambdas, energies) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; translate = true, χenv = χenv);
file = jldopen("test_SU/1_SU", "w")
file["energies"] = energies
close(file)

#=
(psi, lambdas, energies) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; translate = true, χenv = χenv);
file = jldopen("test_SU/2_SU", "w")
file["energies"] = energies
close(file)
(psi, energies) = do_CTMRG(psi, H, ctm_alg, χenv)
file = jldopen("test_SU/2_CTMRG", "w")
file["energies"] = energies
close(file)
(psi, lambdas, energies) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; translate = true, χenv = χenv);
file = jldopen("test_SU/3_SU", "w")
file["energies"] = energies
close(file)
(psi, energies) = do_CTMRG(psi, H, ctm_alg, χenv)
file = jldopen("test_SU/3_CTMRG", "w")
file["energies"] = energies
close(file)
(psi, lambdas, energies) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; translate = true, χenv = χenv);
file = jldopen("test_SU/4_SU", "w")
file["energies"] = energies
close(file)
(psi, energies) = do_CTMRG(psi, H, ctm_alg, χenv)
file = jldopen("test_SU/4_CTMRG", "w")
file["energies"] = energies
close(file)
(psi, lambdas, energies) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg, Js; translate = true, χenv = χenv);
file = jldopen("test_SU/5_SU", "w")
file["energies"] = energies
close(file)
=#