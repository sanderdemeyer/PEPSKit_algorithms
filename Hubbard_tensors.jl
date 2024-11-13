function HubbardSpaces(charge, spin, Ds; P=1, Q=1)
    if charge == "U1" 
        if spin == "U1" # checked and correct
            I = fℤ₂ ⊠ U1Irrep ⊠ U1Irrep
            Ps = Vect[I]((0, -P, 0) => 1, (1, Q-P, 1) => 1, (1, Q-P, -1) => 1, (0, 2*Q-P, 0) => 1)
        elseif spin == "SU2"
            I = fℤ₂ ⊠ U1Irrep ⊠ SU2Irrep
            Ps = Vect[I]((0, -P, 0) => 1, (1, Q-P, 1 // 2) => 1, (0, 2*Q-P, 0) => 1)
        elseif spin == nothing # checked and correct
            I = fℤ₂ ⊠ U1Irrep
            Ps = Vect[I]((0, -P) => 1, (1, Q-P) => 2, (0, 2*Q-P) => 1)
        end
    elseif charge == nothing
        if spin == "U1" # checked and correct
            I = fℤ₂ ⊠ U1Irrep
            Ps = Vect[I]((0, 0) => 2, (1, 1) => 1, (1, -1) => 1)
            # Vs = Vect[I]((0, 0) => Ds[1], (1, 1) => Ds[2], (1, -1) => Ds[3])
        elseif spin == "SU2"
            I = fℤ₂ ⊠ SU2Irrep
            Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 2)
            # Vs = Vect[I]((0, 0) => Ds[1], (1, 1 // 2) => Ds[2])
        elseif spin == nothing # checked and correct
            I = fℤ₂
            Ps = Vect[I]((0) => 2, (1) => 2)
            # Vs = Vect[I]((0) => Ds[1], (1) => Ds[2])
        end
    end
    return I, Ps
end

function rescale_bond_dimensions(input_list::Vector{Int64}, Dmax::Int64)
    naive_list = input_list .* (Dmax/sum(input_list))
    D_list = zeros(Int, length(input_list))
    for (i,e) in enumerate(naive_list)
        D_list[i] = floor(e)
        naive_list[i] -= floor(e)
    end
    missing_D = Dmax - sum(D_list)
    while missing_D > 0
        max_value, index = findmax(naive_list)
        D_list[index] += 1
        naive_list[index] -= 1
        missing_D -= 1
    end
    return D_list
end

function HubbardVirtualSpaces(charge, spin, L, Dmax::Int64; P = 1, Q = 1)
    I, pspace = HubbardSpaces(charge, spin, 0; P = P, Q = Q)
    Ps = fill(pspace, L)
    Vmax_base = Vect[I]
    if charge == "U1"
        if spin == "U1"
            loops = [0:1, -(L*P):1:(L*P), -3:1:3]
            trivial = (0,0,0)
        elseif spin == "SU2"
            loops = [0:1, -(L*P):1:(L*P), 0:1//2:3//2]
            trivial = (0,0,0)
        elseif spin == nothing
            loops = [0:1, -(L*P):1:(L*P)]
            trivial = (0,0)
        end
    elseif charge == nothing
        if spin == "U1"
            loops = [0:1, -3:1:3]
            trivial = (0,0)
        elseif spin == "SU2"
            loops = [0:1, 0:1//2:3//2]
            trivial = (0,0)
        elseif spin == nothing
            return fill(Vect[I](0 => floor(Dmax/2), 1 => floor(Dmax/2)), L)
        end
    end

    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    Vmax = Vmax_base(trivial=>1)     # find maximal virtual space

    if (charge == nothing) && (spin == nothing)
        for a = loops
            Vmax = Vmax_base(a => Dmax) ⊕ Vmax
        end
    else
        for a in Iterators.product(loops...)
            Vmax = Vmax_base(a => Dmax) ⊕ Vmax
        end
    end

    V_max = copy(V)      # if no copy(), V will change along when V_max is changed
    for i in 1:length(V_right)
        V_max[i] = Vmax
    end
    println("V_max = $(V_max), done")
    println("V = $(V), done")

    V_trunc = TensorKit.infimum.(V,V_max)
    println("V_trunc = $(V_trunc), done")
    vspaces = copy(V_trunc)
    for (i,vsp) in enumerate(V_trunc)
        dict = vsp.dims
        number_of_spaces = length(dict)
        println(dict.values)
        println(Dmax)
        println(typeof(dict.values))
        Ds = rescale_bond_dimensions(dict.values, Dmax-1)
        keys = push!(dict.keys, trivial)
        push!(Ds, 1)
        println((Ds))
        new_dict = Dict(sp => D for (sp,D) in zip(keys,Ds))
        vspaces[i] = Vmax_base(new_dict)
    end

    return [vspaces[mod1(i + j - 1,L)] for i in 1:L, j in 1:L]
end

function HubbardOSInteraction(charge, spin; P=1, Q=1)
    I, Ps = HubbardSpaces(charge, spin, 0; P=1, Q=1)
    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)

    if charge == "U1"
        if spin == "U1"
            blocks(onesite)[I((0, 2*Q-P, 0))] .= 1
        elseif spin == "SU2"
            blocks(onesite)[I((0, 2*Q-P, 0))] .= 1
        elseif spin == nothing
            blocks(onesite)[I((0, 2*Q-P))] .= 1
        end
    elseif charge == nothing
        if spin == "U1"
            blocks(onesite)[I((0,0))] .= [0 0; 0 1]
        elseif spin == "SU2"
            blocks(onesite)[I((0,0))] .= [0 0; 0 1]
        elseif spin == nothing
            blocks(onesite)[I((0))] .= [0 0; 0 1]
        end
    end
    return onesite
end

function HubbardHopping(charge, spin; P=1, Q=1)
    I, Ps = HubbardSpaces(charge, spin, 0; P=1, Q=1)
    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)

    if charge == "U1"
        if spin == "U1"
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
        elseif spin == "SU2"
            Vs = Vect[I]((1, Q, 1 // 2) => 1)

            c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
            blocks(c⁺)[I((1, Q-P, 1 // 2))] .= 1
            blocks(c⁺)[I((0, 2*Q-P, 0))] .= sqrt(2)
    
            c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
            blocks(c)[I((1, Q-P, 1 // 2))] .= 1
            blocks(c)[I((0, 2*Q-P, 0))] .= sqrt(2)
    
            @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]

            @tensor n[-1; -2] := c[2 1; -2] * c⁺[-1; 1 2]
        elseif spin == nothing
            firstmethod = false
            if firstmethod
                Vs = Vect[I]((1, Q) => 1)

                c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
                blocks(c⁺)[I((1, Q-P))] .= 1
                blocks(c⁺)[I((0, 2*Q-P))] .= sqrt(2)
        
                c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
                blocks(c)[I((1, Q-P))] .= 1
                blocks(c)[I((0, 2*Q-P))] .= sqrt(2)
        
                @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]     
                
                @tensor n[-1; -2] := c[2 1; -2] * c⁺[-1; 1 2]
            else
                Vodd = Vect[I]((1, Q) => 1)
    
                c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
                blocks(c⁺u)[I((1, Q-P))] .= [1; 0;;]
                blocks(c⁺u)[I((0, 2*Q-P))] .= [0 1]
            
                c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
                blocks(c⁺d)[I((1, Q-P))] .= [0; 1;;]
                blocks(c⁺d)[I((0, 2*Q-P))] .= [-1 0]
                    
                cu = permute(c⁺u', ((2,1), (3,)))
                cd = permute(c⁺d', ((2,1), (3,)))
            
                @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
                @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
                twosite = twosite_up + twosite_down

                @tensor nup[-1; -2] := cu[2 1; -2] * c⁺u[-1; 1 2]
                @tensor ndown[-1; -2] := cd[2 1; -2] * c⁺d[-1; 1 2]
                n = nup + ndown    
            end
        end
    elseif charge == nothing
        if spin == "U1"            
            Vup = Vect[I]((1, 1) => 1)
            Vdown = Vect[I]((1, -1) => 1)
        
            c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vup)
            blocks(c⁺u)[I((1, 1))] .= [1 0]
            blocks(c⁺u)[I((0, 0))] .= [0; 1;;]
            
            c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vdown)
            println(convert(Array, blocks(c⁺d)[I((1, -1))]))
            println(convert(Array, blocks(c⁺d)[I((0, 0))]))
            blocks(c⁺d)[I((1, -1))] .= [1 0]
            blocks(c⁺d)[I((0, 0))] .= [0; -1;;]
        
            cu = permute(c⁺u', ((2,1), (3,)))
            cd = permute(c⁺d', ((2,1), (3,)))

            @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
            @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
            twosite = twosite_up + twosite_down

            @tensor nup[-1; -2] := cu[2 1; -2] * c⁺u[-1; 1 2]
            @tensor ndown[-1; -2] := cd[2 1; -2] * c⁺d[-1; 1 2]
            n = nup + ndown
        elseif spin == "SU2"
            @error "TBA"
        elseif spin == nothing
            Vodd = Vect[I]((1) => 1)

            c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
            blocks(c⁺u)[I((1))] .= [1 0; 0 0]
            blocks(c⁺u)[I((0))] .= [0 0; 0 1]
        
            c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vodd)
            blocks(c⁺d)[I((1))] .= [0 0; 1 0]
            blocks(c⁺d)[I((0))] .= [0 0; -1 0]
                
            cu = permute(c⁺u', ((2,1), (3,)))
            cd = permute(c⁺d', ((2,1), (3,)))
        
            @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
            @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
            twosite = twosite_up + twosite_down

            @tensor nup[-1; -2] := cu[2 1; -2] * c⁺u[-1; 1 2]
            @tensor ndown[-1; -2] := cd[2 1; -2] * c⁺d[-1; 1 2]
            n = nup + ndown
        end
    end
    return twosite, n
end

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