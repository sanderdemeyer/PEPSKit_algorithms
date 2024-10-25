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
    io = open("Hubbard_t_1_U_0.csv", "w")
    mkdir("Hubbard_t_1_U_0_tensors")
    result = fixedpoint(psi_init, h, opt_alg, env_init)
    writedlm(io, [1 result.E norm(result.grad)])
    close(io)
    file = jldopen("Hubbard_t_1_U_0_tensors/1.jld2", "w")
    file["grad"] = copy(result.grad)
    file["E"] = result.E
    file["psi"] = result.peps
    file["grad"] = result.grad
    file["norm_grad"] = norm(result.grad)
    close(file)

    for i = 2:maxiter
        result = fixedpoint(result.peps, h, opt_alg, result.env)
        println("E = $(result.E), grad = $(result.grad)")
        writedlm(io, [i result.E norm(result.grad)])
        close(io)
        file = jldopen("Hubbard_t_1_U_0_tensors/$(i).jld2", "w")
        file["grad"] = copy(result.grad)
        file["E"] = result.E
        file["psi"] = result.peps
        file["grad"] = result.grad
        file["norm_grad"] = norm(result.grad)
        close(file)
    end
    close(io)
end

t = 1
U = 0
T = ComplexF64

lattice_size = 2

I, pspace = ASymSpace()

lattice = fill(pspace, lattice_size, lattice_size)

c⁺c⁻, nup, ndown = ASym_Hopping()
twosite_operator = -t*(c⁺c⁻ + c⁺c⁻')
onsite_operator = U*ASym_OSInteraction()

h = nearest_neighbour_hamiltonian(lattice, twosite)

D = 2
χ = 4

vspace = Vect[I]((0) => D/2, (1) => D/2)
vspace_env = Vect[I]((0) => χ/2, (1) => χ/2)

Pspaces = fill(Ps, lattice_size, lattice_size)
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
    optimizer=LBFGS(4; maxiter=1, gradtol=1e-4, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

# env_init = leading_boundary(env0, psi_init, ctm_alg);

maxiter = 100
result = iterate(psi_init, h, opt_alg, env0, maxiter)
# result = iterate(psi_init, h, opt_alg, env_init, maxiter)
