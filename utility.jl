using TensorKit, PEPSKit, KrylovKit, OptimKit

function get_gate(dτ, Js)
    (Jx, Jy, Jz) = Js
    physical_space = ComplexSpace(2)
    lattice = fill(physical_space, 1, 1)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return exp(-dτ*H)
end

function get_gate_Hubbard(dτ, twosite)
    return exp(-dτ*(twosite))
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
    for i = 1:max_iterations+1
        if (i == max_iterations+1)
            @warn("Not converged after $(max_iterations) iterations. Norm differences are $(norm_dif[1]) and $(norm_dif[2])")
            @tensor A[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
            @tensor B[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
        end

        @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
        @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
        # @tensor Ã[-1; -2 -3 -4 -5] = X[-2 -4 -5; 1] * ãR[1; -1 -3]
        # @tensor B̃[-1; -2 -3 -4 -5] = b̃L[-1 -5; 1] * Y[1; -2 -3 -4]

        ãRnew = update_Ã(envs, X, aR, B, B̃, U)
        b̃Lnew = update_B̃(envs, Y, bL, A, Ã, U)

        # Norm = get_norm_tensor_ver(envs, X, Y)
        norm_dif = (norm(ãRnew - ãR), norm(b̃Lnew - b̃L))

        @tensor Atestnew[-1; -2 -3 -4 -5] := X[-2 -3 -5; 1] * ãRnew[1; -1 -4]
        @tensor Btestnew[-1; -2 -3 -4 -5] := b̃Lnew[-1 -2; 1] * Y[1; -3 -4 -5]
        @tensor Atest[-1; -2 -3 -4 -5] := X[-2 -3 -5; 1] * ãR[1; -1 -4]
        @tensor Btest[-1; -2 -3 -4 -5] := b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
        norm_diftest = (norm(Atestnew - Atest), norm(Btestnew - Btest))
        println("norm_diftest = $(norm_diftest)")
    
        @tensor Anew[-1; -2 -3 -4 -5] := X[-2 -3 -5; 1] * ãRnew[1; -1 -4]
        @tensor Bnew[-1; -2 -3 -4 -5] := b̃Lnew[-1 -2; 1] * Y[1; -3 -4 -5]

        (ãRnew, b̃Lnew, Xnew, Ynew) = update_tensor_with_gauge(envs, X, Y, ãRnew, b̃Lnew)

        @tensor Anew2[-1; -2 -3 -4 -5] := Xnew[-2 -3 -5; 1] * ãRnew[1; -1 -4]
        @tensor Bnew2[-1; -2 -3 -4 -5] := b̃Lnew[-1 -2; 1] * Ynew[1; -3 -4 -5]

        new_normdif = (norm(Atest - Anew2), norm(Btest - Bnew2))
        println("new - norm_dif = $(new_normdif)")


        norm_dif3 = (norm(Anew - Anew2), norm(Bnew - Bnew2))
        println("norm_dif3 = $(norm_dif3)")
        norm_dif2 = (norm(Anew - Ã), norm(Bnew - B̃))



        println("norm_dif = $(norm_dif). norm-diftest = $(norm_diftest)")
        println("norm_dif2 = $(norm_dif2)")
        if (norm_dif[1] < 1e-5) && (norm_dif[2] < 1e-5)
            # @tensor A[-1; -2 -3 -4 -5] = X[-2 -3 -5; 1] * ãR[1; -1 -4]
            # @tensor B[-1; -2 -3 -4 -5] = b̃L[-1 -2; 1] * Y[1; -3 -4 -5]
            println("Converged after $(i) iterations")
            return (A, B)
        else
            # println("norm difference for aR = $(norm(ãRnew - ãR)). Norm difference for bL = $(norm(b̃Lnew - b̃L))")
            ãR = ãRnew
            b̃L = b̃Lnew
            A = Ã
            B = B̃
            X = Xnew
            Y = Ynew
        end
    end
    return (A, B)
end
