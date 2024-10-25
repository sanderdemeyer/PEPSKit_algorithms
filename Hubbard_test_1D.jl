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

function find_gs_half_filling(U, pspace, vspace, lattice_size, twosite_operator, onsite_operator, n; max_iterations = 100)
    Hopping_term = @mpoham sum(twosite_operator{i,i+1} for i in vertices(InfiniteChain(lattice_size)))
    Interaction_term = @mpoham sum(onsite_operator{i} for i in vertices(InfiniteChain(lattice_size)))
    number_term = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

    chem_pot = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

    H_base = -t*Hopping_term + U*Interaction_term

    mps = InfiniteMPS(fill(pspace, lattice_size),fill(vspace, lattice_size))


    mu_old = 0.0
    H = H_base - mu_old*chem_pot
    (mps_old,envs,_) = find_groundstate(mps,H,VUMPS(; maxiter = 20, tol = 10^(-5)))
    filling_old = expectation_value(mps_old, number_term)/2
    E = expectation_value(mps_old, H)

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


t = 1
U = 0 # => E = -1.62, check met Mortier et al 2023
mu = 10
# U1 ipv SU2 for spin
# gmres to arnoldi in svdsolve
# without symmetries?
# low bond dimension wo symmetries

T = ComplexF64

lattice_size = 2

I, pspace = ASymSpace()

lattice = fill(pspace, lattice_size, lattice_size)

twosite, nup, ndown = ASym_Hopping()

cunew = permute(c⁺u', ((2,1), (3,)))
# cunew = permute(cunew, ((3, 1), (2,)))
c⁺c⁻, nup, ndown = ASym_Hopping()
twosite_operator = (c⁺c⁻ + c⁺c⁻')
onsite_operator = ASym_OSInteraction()


# particle_symmetry = Trivial
# spin_symmetry = Trivial
# twosite_operator = e_plusmin(T, particle_symmetry, spin_symmetry) + e_minplus(T, particle_symmetry, spin_symmetry)
# onsite_operator = e_number_updown(T, particle_symmetry, spin_symmetry)
# n = e_number(T, particle_symmetry, spin_symmetry)

D_start = 40



vspace = Vect[I]((0) => D_start/2, (1) => D_start/2)

Hopping_term = @mpoham sum(twosite_operator{i,i+1} for i in vertices(InfiniteChain(lattice_size)))
Interaction_term = @mpoham sum(onsite_operator{i} for i in vertices(InfiniteChain(lattice_size)))
n = nup + ndown
number_term = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

chem_pot = @mpoham sum(n{i} for i in vertices(InfiniteChain(lattice_size)))

energies = []
mus = []
fillings = []
for U in [1.0*i for i = 0:0]
    println("Started for U = $(U)")
    mps, E, mu, filling = find_gs_half_filling(U, pspace, vspace, lattice_size, twosite_operator, onsite_operator, n)
    
    push!(energies, E)
    push!(mus, mu)
    push!(fillings, filling)
end

file = jldopen("test_Hubbard_1D_t_1_U_0to10_D_40_with_signs.jld2", "w")
file["energies"] = energies
file["mus"] = mus
file["fillings"] = fillings
close(file)