using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit

include("Hubbard_tensors.jl")
include("spin_tensors.jl")

P = 4
Q = 2
spin = false

t = 0.001
U = -5

T = ComplexF64

lattice_size = 2

I, pspace = SymSpace(P,Q,spin)

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻ = Hopping(P,Q,spin)
twosite = t*(c⁺c⁻ + c⁺c⁻')
onsite = U*OSInteraction(P,Q,spin)

h = nearest_neighbour_hamiltonian(lattice, twosite, onsite)

# Test with Heisenberg

# pspace = ComplexSpace(2)
# lattice = fill(pspace, lattice_size, lattice_size)

# hopping =
# rmul!(S_xx(Trivial, T; spin=1//2), -1) +
# rmul!(S_yy(Trivial, T; spin=1//2), 1) +
# rmul!(S_zz(Trivial, T; spin=1//2), -1)

# h = nearest_neighbour_hamiltonian(lattice, hopping)

# χbond = 2
# Random.seed!(91283219347)

trivspace = Vect[I]((0, 0, 0) => 1)
vspace0 = Vect[I]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
vspace1 = Vect[I]((0, 0, 0) => 1, (0, 1, 0) => 1, (1, 0, 1 // 2) => 1)

@assert (P / 4 == 1)

vspace = Vect[I]((0,-1,0) => 1, (1, Q - 1, 1 // 2) => 1)

Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = fill(vspace, lattice_size, lattice_size)
Espaces = fill(vspace, lattice_size, lattice_size)

psi_init = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)

# psi_init = InfinitePEPS(2, χbond; unitcell = (lattice_size,lattice_size))

# χenv = 8
# env0 = CTMRGEnv(psi_init, ComplexSpace(χenv));
env0 = CTMRGEnv(psi_init, vspace0)

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