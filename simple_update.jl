include("utility.jl")

function get_norm(psi)
    summation = 0
    for index in [(1,1), (1,2), (2,1), (2,2)]
        summation += norm(psi[index...])^2
    end
    println("norm = $(summation)")
end

function check_norm(psi)
    if abs(norm(psi) - 1) > 1e-10
        throw("not normalized")
    end
    # if norm(psi[1,1]) < 1e-10
    #     println("1, 1: norm = 0")
    #     throw("stop")
    # elseif norm(psi[1,2]) < 1e-10
    #     println("1, 2: norm = 0")
    #     throw("stop")
    # elseif norm(psi[2,1]) < 1e-10
    #     println("2, 1: norm = 0")
    #     throw("stop")
    # elseif norm(psi[2,2]) < 1e-10
    #     println("2, 2: norm = 0")
    #     throw("stop")
    # end
end
function gauge_fix_north(psi, lambdas)
    check_norm(psi)
    # psi = InfinitePEPS(2, 2; unitcell=(2,2))
    # lambdas = fill(id(ℂ^2),8)

    @tensor ML[-2; -1] := psi[1,1][7; 1 -1 3 5] * conj(psi[1,1][7; 8 -2 9 10]) * 
    lambdas[1][1; 2] * lambdas[8][4; 3] * lambdas[3][6; 5] *
    conj(lambdas[1][8; 2]) * conj(lambdas[8][4; 9]) * conj(lambdas[3][6; 10])

    @tensor MR[-1; -2] := psi[1,2][7; 1 3 5 -1] * conj(psi[1,2][7; 8 9 10 -2]) * 
    lambdas[5][1; 2] * lambdas[3][3; 4] * lambdas[4][6; 5] * 
    conj(lambdas[5][8; 2]) * conj(lambdas[3][9; 4]) * conj(lambdas[4][6; 10])

    uL, dL, VL = tsvd(ML)
    uR, dR, VR = tsvd(MR)
    # ((UL, SL, VL), (UR, SR, VR)) = tsvd.([ML, MR])

    if (norm(VL' - uL) > 1e-10) || (norm(VR' - uR) > 1e-10)
        @warn "Diagonalization failed"
    end

    λ′ = sqrt(dL) * uL' * lambdas[2] * uR * sqrt(dR)
    wL, λ, wR = tsvd(λ′) # \lambda = \Tilde{\lambda} in the paper

    # if (norm(wR' - wL) > 1e-10)
    #     @warn "Something may have gone wrong, but please ignore this"
    # end

    x = wL' * sqrt(dL) * uL'
    y = uR * sqrt(dR) * wR'

    @tensor psi11_new[-1; -2 -3 -4 -5] := psi[1,1][-1; -2 1 -4 -5] * inv(x)[1; -3]
    @tensor psi12_new[-1; -2 -3 -4 -5] := psi[1,2][-1; -2 -3 -4 1] * inv(y)[-5; 1]

    normalization = sqrt((norm(psi[1,1])^2+norm(psi[1,2])^2)/(norm(psi11_new)^2+norm(psi12_new)^2))
    psi[1,1] = copy(normalization*psi11_new)
    psi[1,2] = copy(normalization*psi12_new)

    # @tensor psi[1,1][-1; -2 -3 -4 -5] = psi[1,1][-1; -2 1 -4 -5] * inv(x)[1; -3]
    # @tensor psi[1,2][-1; -2 -3 -4 -5] = psi[1,2][-1; -2 -3 -4 1] * inv(y)[-5; 1]
    lambdas[2] = copy(λ)/norm(λ)
    psi = (1/norm(psi))*psi

    check_norm(psi)

    return (psi, lambdas)
end

function gauge_fix(psi, lambdas)
    # println("first, psi = $(psi[1,1])")
    for i = 1:4
        # println("then, psi = $(psi[1,1])")
        check_norm(psi)
        (psi, lambdas) = gauge_fix_north(psi, lambdas)
        check_norm(psi)
        psi = rotate_psi(psi)
        check_norm(psi)
        lambdas = rotate_lambdas(lambdas)
    end
    psi = translate_psi(psi)
    lambdas = translate_lambdas(lambdas)
    for i = 1:4
        (psi, lambdas) = gauge_fix_north(psi, lambdas)
        psi = rotate_psi(psi)
        lambdas = rotate_lambdas(lambdas)
    end
    return (psi, lambdas)
end
function do_QR(left_t, right_t, left_index, right_index)
    println((Tuple(setdiff(2:5, left_index)), (1, left_index)))

    Ql, R = leftorth(left_t, (Tuple(setdiff(2:5, left_index)), (1, left_index)), alg = QR())
    L, Qr = rightorth(right_t, ((1, right_index), Tuple(setdiff(2:5, right_index))), alg = LQ())
    return (Ql, R, L, Qr)
end

function absorb_lambdas_OLD(tensor, excl_index, lambdas)
    contr_indices_tens = [i == excl_index ? -excl_index : contraction_indices[i-1] for i = 2:5]
    @tensor tensor_new[-1; -2 -3 -4 -5] := tensor[]
    lambda_list = []
    lambda_contraction_list = []
    for (i, lambda) in enumerate(lambdas)
        if i != excl_index - 1
            push!(lambda_list, lambda)
            push!(lambda_contraction_list, )
        end
    end
end

coords = [(1, 1), (2, 1), (2, 2), (1, 2)]
SM = transpose([1 0 0 3; 2 4 0 0; 4 2 0 0; 0 3 1 0; 0 1 3 0; 0 0 4 2; 0 0 2 4; 3 0 0 1])
edges = [(1, 4), (1, 2), (1, 2), (2, 3), (2, 3), (3, 4), (3, 4), (1, 4)]

lambdas_surrounded = [(1, 2, 8, 3), (5, 3, 4, 2), (4, 7, 5, 6), (8, 6, 1, 7)]

contraction_indices = [1, 1, 2, 2] # mod1(i,2)

# for (i,edge) in enumerate(edges)
#     left_t = psi_init[coords[edge[1]]...]
#     right_t = psi_init[coords[edge[2]]...]

#     (left_t) = absorb_lambdas(left_t, SM[edge[1],i]+1, lambdas_surrounded[edge[1]])

#     (Ql, Rl, Lr, Qr) = do_QR(left_t, right_t, SM[edge[1],i]+1, SM[edge[2],i]+1)
#     println("done")
# end


T = TensorMap(randn, ℂ^2 ⊗ ℂ^3, ℂ^4 ⊗ ℂ^5)
Q, R = leftorth(T, (1,2),(3,4), alg = QR())
Q, R = rightorth(T, (1,2),(3,4), alg = LQ())
@tensor T_new[-1 -2; -3 -4] := Q[-1 -2; 1] * R[1; -3 -4]


function absorb_lambdas(left, right, lambdas; inverse = false)
    if inverse
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 -3 2 3] * inv(lambdas[1])[1; -2] * inv(lambdas[8])[-4; 2] * inv(lambdas[3])[-5; 3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 -5] * inv(lambdas[5])[1; -2] * inv(lambdas[3])[2; -3] * inv(lambdas[4])[-4; 3]
    else
        @tensor left_t[-1; -2 -3 -4 -5] := left[-1; 1 -3 2 3] * lambdas[1][1; -2] * lambdas[8][-4; 2] * lambdas[3][-5; 3]
        @tensor right_t[-1; -2 -3 -4 -5] := right[-1; 1 2 3 -5] * lambdas[5][1; -2] * lambdas[3][2; -3] * lambdas[4][-4; 3]
    end
    return (left_t, right_t)
end

function simple_update_north(psi, lambdas, dτ, χ, Js)
    check_norm(psi)
    U = get_gate(dτ, Js)

    left = psi[1,1]
    right = psi[1,2]

    # Absorb lambdas into left and right tensors
    (left_t, right_t) = absorb_lambdas(left, right, lambdas, inverse = false)

    # Group the legs of the tensor and perform QR, LQ decomposition
    (left_index, right_index) = (3, 5)
    
    Ql, R = leftorth(left_t, (Tuple(setdiff(2:5, left_index)), (1, left_index)), alg = QR())
    L, Qr = rightorth(right_t, ((1, right_index), Tuple(setdiff(2:5, right_index))), alg = LQ())

    Ql = permute(Ql, ((4,), (1, 2, 3)))
    R = permute(R, ((2, 3), (1,)))

    @tensor left_new[-1; -2 -3 -4 -5] := Ql[1; -2 -4 -5] * R[-1 -3; 1]
    @tensor right_new[-1; -2 -3 -4 -5] := L[-1 -5; 1] * Qr[1; -2 -3 -4]

    # println("left_new = $(summary(left_new))")
    # println("right_new = $(summary(right_new))")

    if norm(left_new - left_t) > 1e-7
        @warn "norm difference between reconstructed and original tensor after QR decomposition is $(norm(left_new - left_t))"
    end
    if norm(right_new - right_t) > 1e-7
        @warn "norm difference between reconstructed and original tensor after LQ decomposition is $(norm(right_new - right_t))"
    end
    # println(summary(Ql))
    # println(summary(R))
    # println(summary(L))
    # println(summary(Qr))

    @tensor Θ[-1 -2; -3 -4] := R[1 2; -1] * L[4 3; -4] * U[-2 -3; 1 4] * lambdas[2][2; 3]

    # println("theta = $(summary(Θ))")

    (R_new, lambda_new, L_new) = tsvd(Θ, trunc = truncdim(χ))

    # println("start")
    # println(summary(Θ))
    # println(summary(R_new))
    # println(summary(lambda_new))
    # println(summary(L_new))
    # println("end")

    @tensor Plnew[-1; -2 -3 -4 -5] := Ql[1; -2 -4 -5] * R_new[1 -1; -3]
    @tensor Prnew[-1; -2 -3 -4 -5] := L_new[-5; -1 1] * Qr[1; -2 -3 -4]
    # println("Plnew = $(summary(Plnew))")
    # println("Prnew = $(summary(Prnew))")
    # println(summary(Plnew))
    # println(summary(Prnew))

    (left_new, right_new) = absorb_lambdas(Plnew, Prnew, lambdas, inverse = true)

    normalization = sqrt((norm(psi[1,1])^2+norm(psi[1,2])^2)/(norm(left_new)^2+norm(right_new)^2))

    psi[1,1] = normalization * left_new
    psi[1,2] = normalization * right_new
    lambdas[2] = lambda_new / norm(lambda_new)

    check_norm(psi)
    return (psi, lambdas)
end

function rotate_lambdas(lambdas; permuted = "nothing")
    if permuted == "forward"
        return [lambdas[3],
                permute(lambdas[4], ((2,), (1,))),
                permute(lambdas[5], ((2,), (1,))),
                lambdas[6],
                lambdas[7],
                permute(lambdas[8], ((2,), (1,))),
                permute(lambdas[1], ((2,), (1,))),
                lambdas[2]
        ] 
    elseif permuted == "inverse"
        return [permute(lambdas[3], ((2,), (1,))),
                lambdas[4],
                lambdas[5],
                permute(lambdas[6], ((2,), (1,))),
                permute(lambdas[7], ((2,), (1,))),
                lambdas[8],
                lambdas[1],
                permute(lambdas[2], ((2,), (1,)))
        ] 
    elseif permuted == "nothing"
        return [lambdas[3],
            lambdas[4],
            lambdas[5],
            lambdas[6],
            lambdas[7],
            lambdas[8],
            lambdas[1],
            lambdas[2]
        ] 
    end
    error("Invalid argument: permuted")
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

function rotate_psi(psi)
    psi_new = copy(psi)
    psi_new[1,1] = permute(rotl90(psi[1,2]), ((1,), (5, 2, 3, 4)))
    psi_new[1,2] = permute(rotl90(psi[2,2]), ((1,), (5, 2, 3, 4)))
    psi_new[2,2] = permute(rotl90(psi[2,1]), ((1,), (5, 2, 3, 4)))
    psi_new[2,1] = permute(rotl90(psi[1,1]), ((1,), (5, 2, 3, 4)))
    return psi_new
end

function rotate_psi_old(psi)
    psi_new = copy(psi)
    psi_new[1,1] = rotl90(psi[1,2])
    psi_new[1,2] = rotl90(psi[2,2])
    psi_new[2,2] = rotl90(psi[2,1])
    psi_new[2,1] = rotl90(psi[1,1])
    return psi_new
end

function absorb_lambdas_in_peps(psi, lambdas)
    lambdas_sqrt = sqrt.(lambdas)
    psi_new = copy(psi)
    for (coord, sur) in zip(coords, lambdas_surrounded)
        # @tensor psi_new[coord...][-1; -2 -3 -4 -5] = psi[coord...][-1; 1 2 3 4] * lambdas_sqrt[sur[1]][1; -2] * lambdas_sqrt[sur[2]][2; -3] * lambdas_sqrt[sur[3]][3; -4] * lambdas_sqrt[sur[4]][4; -5]
        @tensor psi_new[coord...][-1; -2 -3 -4 -5] = psi[coord...][-1; 1 2 3 4] * lambdas_sqrt[sur[1]][1; -2] * lambdas_sqrt[sur[2]][2; -3] * lambdas_sqrt[sur[3]][-4; 3] * lambdas_sqrt[sur[4]][-5; 4]
    end
    return psi_new
end

function get_energy(psi_base, lambdas, H, ctm_alg, χenv)
    psi_new = absorb_lambdas_in_peps(psi_base, lambdas)
    env0 = CTMRGEnv(psi_new, ComplexSpace(χenv));
    env = leading_boundary(env0, psi_new, ctm_alg);
    return expectation_value(psi, H, env)
end

function translate_psi(psi)
    psi_new = copy(psi)
    psi_new[1,1] = psi[1,2]
    psi_new[1,2] = psi[1,1]
    psi_new[2,1] = psi[2,2]
    psi_new[2,2] = psi[2,1]
    return psi_new
end


function simple_update(psi, H, dτ, χ, max_iterations, ctm_alg; χenv = 3*χ, translate = false)
    lambdas = fill(id(ℂ^χ),8)

    (psi, lambdas) = gauge_fix(psi, lambdas)
    for i = 1:max_iterations
        # println("Started with iteration i = $(i)")
        psi_old = copy(psi)
        for i = 1:4
            check_norm(psi)
            (psi, lambdas) = gauge_fix(psi, lambdas)
            check_norm(psi)
            (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, Js)
            check_norm(psi)
            psi = rotate_psi(psi)
            lambdas = rotate_lambdas(lambdas)
        end
        if translate
            psi = translate_psi(psi)
            lambdas = translate_lambdas(lambdas)
            for i = 1:4
                (psi, lambdas) = gauge_fix(psi, lambdas)
                (psi, lambdas) = simple_update_north(psi, lambdas, dτ, χ, Js)
                psi = rotate_psi(psi)
                lambdas = rotate_lambdas(lambdas)
            end
        end
        # println("change in norm is $(norm(psi-psi_old))")
        if mod(i, 5) == 0
            check_norm(psi)
            energy = get_energy(deepcopy(psi), lambdas, H, ctm_alg, χenv)
            check_norm(psi)
            println("Energy after SU is $(energy)")
            psi = 1/(norm(psi)) * psi
            check_norm(psi)
        end
    end
    return (psi, lambdas)
end

function do_CTMRG(psi, H, ctm_alg, χenv)

    opt_alg = PEPSOptimize(;
        boundary_alg=ctm_alg,
        optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
        gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
        reuse_env=true,)
        
    env0 = CTMRGEnv(psi, ComplexSpace(χenv));
    env_init = leading_boundary(env0, psi, ctm_alg);
    result = fixedpoint(psi, H, opt_alg, env_init)    
    println("Energy after SU is $(result.E)")
    return  result.peps
end

dτ = 0.1
χ = 2
D = 2
χenv = 2

unitcell = (2, 2)
max_iterations = 50

Js = (-1, 1, -1)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

H = square_lattice_heisenberg(; Jx = Js[1], Jy = Js[2], Jz = Js[3], unitcell = (2,2))

psi = InfinitePEPS(D, D; unitcell)
psi = (1/norm(psi))*psi

psi = do_CTMRG(psi, H, ctm_alg, χenv)
(psi, lambdas) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg; translate = true);
psi = do_CTMRG(psi, H, ctm_alg, χenv)
(psi, lambdas) = simple_update(psi, H, dτ, χ, max_iterations, ctm_alg; translate = true);

println("Done")

# psi_rot = rotate_psi(psi_init);
# psi_rot11 = permute(psi_rot[1,1], ((1,), (5, 2, 3, 4)));