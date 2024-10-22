using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit
using DelimitedFiles

include("Hubbard_tensors.jl")
include("spin_tensors.jl")

function iterate(psi_init, h, opt_alg, env_init, maxiter)
    io = open("check_energies_Hubbard_ASym.csv", "w")
    result = fixedpoint(psi_init, h, opt_alg, env_init)
    println("E = $(result.E), grad = $(result.grad)")
    writedlm(io, [1 result.E result.grad])
    for i = 2:maxiter
        result = fixedpoint(result.peps, h, opt_alg, result.env)
        println("E = $(result.E), grad = $(result.grad)")
        writedlm(io, [i result.E result.grad])
    end
    close(io)
end

t = 1
U = 0 # => E = -1.62, check met Mortier et al 2023

# U1 ipv SU2 for spin
# gmres to arnoldi in svdsolve
# without symmetries?
# low bond dimension wo symmetries

T = ComplexF64

lattice_size = 2

I, pspace = ASymSpace()

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻, nup, ndown = ASym_Hopping()
twosite = -t*(c⁺c⁻ + c⁺c⁻')
onsite = U*ASym_OSInteraction()

# P = 1
# Q = 1
# spin = true
# Isym, pspacesym = SymSpace(P, Q, spin)
# c⁺c⁻sym, nsym = Hopping(P, Q, spin)
# onsitesym = U*OSInteraction(P, Q, spin)

h = nearest_neighbour_hamiltonian(lattice, twosite, onsite)


D = 2
χ = 8

vspace = Vect[I]((0) => D/2, (1) => D/2)
vspace_env = Vect[I]((0) => χ/2, (1) => χ/2)

Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = Espaces = fill(vspace, lattice_size, lattice_size)

psi_init = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env0 = CTMRGEnv(psi_init, vspace_env)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=1, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

# env_init = leading_boundary(env0, psi_init, ctm_alg);

maxiter = 200
result = iterate(psi_init, h, opt_alg, env0, maxiter)
# result = iterate(psi_init, h, opt_alg, env_init, maxiter)
