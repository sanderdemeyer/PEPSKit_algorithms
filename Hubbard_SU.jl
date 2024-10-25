include("utility.jl")
include("Hubbard_tensors.jl")
using JLD2

dτ = 1e-4
D = 2
χ = D
χenv = 40

lattice_size = 2

unitcell = (lattice_size, lattice_size)
max_iterations = 30000

t = 1.0
U = 0.0

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

I, pspace = ASymSpace()

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻ = ASym_Hopping()
twosite_operator = -t*(c⁺c⁻ + c⁺c⁻')
onsite_operator = U*ASym_OSInteraction()

H = nearest_neighbour_hamiltonian(lattice, twosite_operator)

vspace = Vect[I]((0) => D/2, (1) => D/2)
vspace_env = Vect[I]((0) => χenv/2, (1) => χenv/2)

Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = Espaces = fill(vspace, lattice_size, lattice_size)

psi = normalize(InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces))
env0 = CTMRGEnv(psi, vspace_env)

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


function simple_update_north(psi, lambdas, dτ, χ, twosite_operator, base_space)
    U = get_gate_Hubbard(dτ, twosite_operator)

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

function get_energy(psi, H, ctm_alg, χenv)
    vspace_env = Vect[I]((0) => χenv/2, (1) => χenv/2)
    env0 = CTMRGEnv(psi, vspace_env);
    env = leading_boundary(env0, psi, ctm_alg);
    return expectation_value(psi, H, env)
end

function simple_update(psi, H, dτ, vspace, max_iterations, ctm_alg, twosite_operator; χenv = 3*χ, translate = false)
    lambdas = fill(id(vspace),8)
    base_space = psi[1,1].dom[2]

    energies = []
    # Do gauge fix
    for i = 1:max_iterations
        println("Iteration i = $(i)")
        for i = 1:4
            (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, twosite_operator, base_space)
            psi = rotate_psi_l90(psi)
            lambdas = rotate_lambdas_l90(lambdas)
        end
        psi = translate_psi_diag(psi)
        lambdas = translate_lambdas_diag(lambdas)
        for i = 1:4
            (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, twosite_operator, base_space)
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
    @error "fix to Hubbard"
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

(psi, lambdas, energies) = simple_update(psi, H, dτ, vspace, max_iterations, ctm_alg, twosite_operator; translate = true, χenv = χenv);

file = jldopen("Hubbard_t_1_U_0_SU.jld2", "w")
file["energies"] = energies
close(file)