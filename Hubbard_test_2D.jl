using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit
using DelimitedFiles
using JLD2

include("Hubbard_tensors.jl")

function custom_finalize_base(name, x, f, g, numiter)
    file = jldopen(name*"/$(numiter)", "w")
    file["psi"] = copy(x)
    file["E"] = f
    file["grad"] = g
    file["norm_grad"] = norm(grad)
    close(file)
    return x, f, g
end
    
t = 1
U = 0
P = Q = 1
charge = nothing
spin = nothing

D = 2
χenv = 4 # Yuchi uses 3D^2

lattice_size = 2

I, pspace = HubbardSpaces(charge, spin, 0; P = P, Q = Q)

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻, n = HubbardHopping(charge, spin; P = P, Q = Q)
twosite_operator = -t*(c⁺c⁻ + c⁺c⁻')
onsite_operator = U*HubbardOSInteraction(charge, spin; P = P, Q = Q)

h = nearest_neighbour_hamiltonian(lattice, twosite_operator)


# vspace = Vect[I]((0) => D/2, (1) => D/2)
# vspace_env = Vect[I]((0) => χenv/2, (1) => χenv/2)

vspaces = HubbardVirtualSpaces(charge, spin, lattice_size, D; P = P, Q = Q)
vspaces_env = HubbardVirtualSpaces(charge, spin, lattice_size, χenv; P = P, Q = Q)

Pspaces = fill(pspace, lattice_size, lattice_size)
Nspaces = Espaces = vspaces

psi_init = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env0 = CTMRGEnv(psi_init, vspaces_env)

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
    optimizer=LBFGS(4; maxiter=1, gradtol=1e-4, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

opt_alg_geomsum = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=1, gradtol=1e-4, verbosity=2),
    gradient_alg=GeomSum(; tol=1e-8, iterscheme=:fixed),
    reuse_env=true,
)

# opt_alg = PEPSOptimize(;
#     boundary_alg=ctm_alg,
#     optimizer=ConjugateGradient(; maxiter=2, gradtol=1e-4, verbosity=2),
#     gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
#     reuse_env=true,
# )

env_init = leading_boundary(env0, psi_init, ctm_alg);
maxiter = 100
name = "Hubbard_t_$(t)_U_$(U)_tensors_D_$(D)_chi_$(χenv)_charge_$(charge)_spin_$(spin)"

custom_finalize! = (x, f, g, numiter) -> custom_finalize_base(name, x, f, g, numiter)

mkdir(name)
result = fixedpoint(psi_init, h, opt_alg_geomsum, env_init; finalize! = custom_finalize!)