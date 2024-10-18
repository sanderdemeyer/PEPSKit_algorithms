using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit

include("Hubbard_tensors.jl")
include("spin_tensors.jl")

P = 2
Q = 2
spin = false

t = 1
U = 5

T = ComplexF64

lattice_size = 2

@assert (P % 2 == 0)

I, pspace = SymSpace(P,Q,spin)

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻ = Hopping(P,Q,spin)
twosite = t*(c⁺c⁻ + c⁺c⁻')
onsite = U*OSInteraction(P,Q,spin)

h = nearest_neighbour_hamiltonian(lattice, twosite, onsite)


trivspace = Vect[I]((0, 0, 0) => 1)
vspace0 = Vect[I]((0, 0, 0) => 1, (1, Q, 1 // 2) => 1, (0, 2*Q, 1) => 1)
vspace1 = Vect[I]((0, -div(P,2), 0) => 1, (1, Q-div(P,2), 1 // 2) => 1, (0, 2*Q - div(P,2), 1) => 1)

vspace_env = Vect[I]((0, 0, 0) => 1, (1, Q, 1 // 2) => 1, (0, 2*Q, 1) => 1,
                    (0, -div(P,2), 0) => 1, (1, Q-div(P,2), 1 // 2) => 1, (0, 2*Q - div(P,2), 1) => 1
)

spacecheck = vspace0 ⊗ vspace0' ⊗ vspace1 ⊗ vspace1'

Pspaces = fill(pspace, lattice_size, lattice_size)

# Nspaces = fill(vspace00, lattice_size, lattice_size)
# Espaces = fill(vspace10, lattice_size, lattice_size)
Nspaces = [vspace10 vspace0; vspace0 vspace1]
Espaces = [vspace10 vspace0; vspace0 vspace1]

psi_init = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)

vspace_NW = Vect[I]((0, 0, 0) => 1)
vspace_NE = Vect[I]((0, 0, 0) => 1, (0, Q, 1 // 2) => 1, (0, 2*Q, 1) => 1)
vspace_SE = Vect[I]((0, 0, 0) => 1, (0, 2*Q, 0) => 1, (0, 2*Q, 1) => 1,
                    (0, Q, 1 // 2) => 1)
vspace_SW = Vect[I]((0, 0, 0) => 1, (0, 2*Q, 0) => 1, (0, 2*Q, 1) => 1,
                    (0, Q, 1 // 2) => 1, (0, 0, 1) => 1)

                    vspace_corner0 = Vect[I]((0, 0, 0) => 1, (0, 2*Q, 0) => 1, (0, 2*Q, 1) => 1,
                    (0, Q, 1 // 2) => 1, (0, 0, 1) => 1)


dict_env = ((0, 0, 0) => 1, (0, 2*Q, 0) => 1, (0, 2*Q, 1) => 1,
(0, Q, 1 // 2) => 1, (0, 0, 1) => 1,
(0, -div(P,2), 0) => 1, (0, 2*Q - div(P,2), 0) => 1, (0, 2*Q - div(P,2), 1) => 1,
(0, Q - div(P,2), 1 // 2) => 1, (0, -div(P,2), 1) => 1,(0, -div(P,2), 0) => 1, 
(0, 2*Q - P, 0) => 1, (0, 2*Q - P, 1) => 1,
(0, Q - P, 1 // 2) => 1, (0, -P, 1) => 1)

vspace_env = Vect[I](unique(dict_env))





# psi_init = InfinitePEPS(2, χbond; unitcell = (lattice_size,lattice_size))

# χenv = 8
# env0 = CTMRGEnv(psi_init, ComplexSpace(χenv));
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
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

env_init = leading_boundary(env0, psi_init, ctm_alg);

result = fixedpoint(psi_init, h, opt_alg, env_init)
println("Energy = $(result.E)")