using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit

println("Started")

include("Hubbard_tensors.jl")

t = 1
U = 0 # => E = -1.62, check met Mortier et al 2023

T = ComplexF64
lattice_size = 1

I, pspace = ASymSpace()

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻, nup, ndown = ASym_Hopping()
twosite = -t*(c⁺c⁻ + c⁺c⁻')
onsite = U*ASym_OSInteraction()

h = nearest_neighbour_hamiltonian(lattice, twosite, onsite)

D = 2
χ = 2
maxiter = 300

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
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=Arnoldi(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=maxiter, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

result = fixedpoint(psi_init, h, opt_alg, env0)