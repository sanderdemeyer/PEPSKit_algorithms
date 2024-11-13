using Random
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit
using MPSKit
using DelimitedFiles
using JLD2

include("Hubbard_tensors.jl")

function iterate(psi_init, h, opt_alg, env_init, maxiter, name)
    mkdir(name)
    result = fixedpoint(psi_init, h, opt_alg, env_init)
    file = jldopen(name*"/1.jld2", "w")
    file["grad"] = copy(result.grad)
    file["E"] = result.E
    file["psi"] = result.peps
    file["grad"] = result.grad
    file["norm_grad"] = norm(result.grad)
    close(file)

    for i = 2:maxiter
        result = fixedpoint(result.peps, h, opt_alg, result.env)
        println("E = $(result.E), grad = $(result.grad)")
        file = jldopen(name*"/$(i).jld2", "w")
        file["grad"] = copy(result.grad)
        file["E"] = result.E
        file["psi"] = result.peps
        file["grad"] = result.grad
        file["norm_grad"] = norm(result.grad)
        close(file)
    end
end

t = 1
U = 0
P = 1
Q = 1
charge = "U1"
spin = nothing

lattice_size = 2

I, pspace = HubbardSpaces(charge, spin, 0; P = P, Q = Q)

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻, n = HubbardHopping(charge, spin; P = P, Q = Q)
twosite_operator = -t*(c⁺c⁻ + c⁺c⁻')
onsite_operator = U*HubbardOSInteraction(charge, spin; P = P, Q = Q)

h = nearest_neighbour_hamiltonian(lattice, twosite_operator)

D = 6
χenv = 12 # Yuchi uses 3D^2

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

# opt_alg = PEPSOptimize(;
#     boundary_alg=ctm_alg,
#     optimizer=LBFGS(4; maxiter=1, gradtol=1e-4, verbosity=2),
#     gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
#     reuse_env=true,
# )

opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=ConjugateGradient(; maxiter=2, gradtol=1e-4, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

# env_init = leading_boundary(env0, psi_init, ctm_alg);

maxiter = 2

name = "Hubbard_t_1_U_0_tensors_D_$(D)_chi_$(χenv)"
result = iterate(psi_init, h, opt_alg, env0, maxiter, name)
# result = iterate(psi_init, h, opt_alg, env_init, maxiter)