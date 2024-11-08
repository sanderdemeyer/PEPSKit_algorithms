using LinearAlgebra
using Random
using TensorKit
using KrylovKit
using OptimKit
using MPSKit
using MPSKitModels
using DelimitedFiles
using JLD2

include("Hubbard_tensors.jl")

function find_gs_half_filling(U, pspaces, vspaces, lattice_size, twosite_operator, onsite_operator, n; max_iterations = 100, charge = nothing, impose_mu_a_priori = true)
    t = 1
    
    Hopping_term = @mpoham sum(twosite_operator{i,i+1} for i in vertices(InfiniteChain(lattice_size)))
    Interaction_term = @mpoham sum(onsite_operator{i} for i in vertices(InfiniteChain(lattice_size)))
    number_term = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

    chem_pot = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

    H_base = -t*Hopping_term + U*Interaction_term

    mps = InfiniteMPS(pspaces,vspaces)

    if impose_mu_a_priori
        mu_old = -U/2
        H = H_base + mu_old*chem_pot
        (mps_old,envs,_) = find_groundstate(mps,H,VUMPS(; maxiter = 20, tol = 10^(-5)))
        filling_old = expectation_value(mps_old, number_term)/2
        E = expectation_value(mps_old, H_base)
        return (E, mu_old, filling_old)
    end

    mu_old = 0.0
    H = H_base - mu_old*chem_pot
    (mps_old,envs,_) = find_groundstate(mps,H,VUMPS(; maxiter = 20, tol = 10^(-5)))
    filling_old = expectation_value(mps_old, number_term)/2
    E = expectation_value(mps_old, H)

    if charge == "U1"
        return (E, mu_old, filling_old)
    end

    # return mps, E, mu_old, filling_old
    mu = 1.0
    H = H_base - mu*chem_pot
    (mps,envs,_) = find_groundstate(mps,H,VUMPS(; maxiter = 20, tol = 10^(-5)))
    filling = expectation_value(mps, number_term)/2

    for i = 1:max_iterations
        if abs(filling - 1) > 1e-2
            mu_new = mu + (1-filling)*(mu-mu_old)/(filling-filling_old)
            mu_old = mu
            filling_old = filling
            mu = mu_new

            H = H_base - mu*chem_pot
            (mps,envs,_) = find_groundstate(mps,H,VUMPS(; maxiter = 20, tol = 10^(-5)))
            filling = expectation_value(mps, number_term)/2
            println("new mu = $(mu_new) gives filling $(filling)")
        else
            println("Converged after $(max_iterations) iterations, mu = $(mu) and f = $(filling)")
            return expectation_value(mps, H_base), mu, filling
        end
    end
    @warn "Not converged after $(max_iterations) iterations, mu = $(mu) and f = $(filling)"
    return expectation_value(mps, H_base), mu, filling
end

charge = "U1"
spin = nothing
P = 1
Q = 1

T = ComplexF64
lattice_size = 2

D_start = 40
L = 2

I, pspace = HubbardSpaces(charge, spin, D_start; P = P, Q = Q)
pspaces = fill(pspace, L)
vspaces = HubbardVirtualSpaces(charge, spin, L, D_start; P = P, Q = Q)

c⁺c⁻,n = HubbardHopping(charge, spin; P=1, Q=1)
twosite_operator = (c⁺c⁻ + c⁺c⁻')
onsite_operator = HubbardOSInteraction(charge, spin; P=1, Q=1)

print("printing the tensors")
println("onsite")
println(onsite_operator)
println("n")
println(n)

Hopping_term = @mpoham sum(twosite_operator{i,i+1} for i in vertices(InfiniteChain(lattice_size)))
Interaction_term = @mpoham sum(onsite_operator{i} for i in vertices(InfiniteChain(lattice_size)))
number_term = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

chem_pot = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

impose_mu_a_priori = !(charge != nothing)

energies = []
mus = []
fillings = []
for U in [1.0*i for i = 0:10]
    println("Started for U = $(U)")

    println(pspaces)
    println(vspaces)
    E, mu, filling = find_gs_half_filling(U, pspaces, vspaces, lattice_size, twosite_operator, onsite_operator, n; charge = charge, impose_mu_a_priori = impose_mu_a_priori)
    
    push!(energies, E)
    push!(mus, mu)
    push!(fillings, filling)
end

file = jldopen("test_Hubbard_1D_D_$(D_start)_charge_$(charge)_spin_$(spin)_imposeapriori_$(impose_mu_a_priori).jld2", "w")
file["energies"] = energies
file["mus"] = mus
file["fillings"] = fillings
close(file)


ene = []
for charge in [nothing "U1"]
    for spin in [nothing "U1"]
        impose_mu_a_priori = !(charge != nothing)
        try
            file = jldopen("test_Hubbard_1D_D_$(40)_charge_$(charge)_spin_$(spin)_imposeapriori_$(impose_mu_a_priori).jld2", "r")
        catch
            file = jldopen("test_Hubbard_1D_D_$(50)_charge_$(charge)_spin_$(spin)_imposeapriori_$(impose_mu_a_priori).jld2", "r")
        end
        energies = file["energies"]
        push!(ene, energies)
        close(file)
    end
end

using Plots
plt = scatter(1:11, real.(ene[1]), label = "fZ2 x not x not")
scatter!(1:11, real.(ene[1]), label = "fZ2 x not x U1")
scatter!(1:11, real.(ene[1]), label = "fZ2 x U1 x not")
scatter!(1:11, real.(ene[1]), label = "fZ2 x U1 x U1")
display(plt)

