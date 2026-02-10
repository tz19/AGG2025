"""
FEniCSx version of AGG (Abnormal Grain Growth) phase-field model.
Migrated from legacy FEniCS (AGG_Ni_001texture.py).
Equations: linear elasticity with orientation-dependent stiffness,
phase-field evolution with free energy and elastic energy coupling.

Run: python Ni_100_FEniCSx.py  (or mpirun -n N python Ni_100_FEniCSx.py)
Requires: dolfinx, ufl, mpi4py, petsc4py, scipy, numpy.
"""

from pathlib import Path
import sys
from mpi4py import MPI
import numpy as np
import math
import ufl
from dolfinx import fem, io, mesh
from dolfinx.mesh import CellType
from dolfinx.fem.petsc import (
    LinearProblem,
    NonlinearProblem,
    create_matrix,
    create_vector,
    assemble_matrix,
    assemble_vector,
)
from petsc4py import PETSc

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# Phase-field form has deep UFL expression tree (16 components, derivatives);
# raise recursion limit so FFCx JIT compilation does not hit it.
if sys.getrecursionlimit() < 8000:
    sys.setrecursionlimit(8000)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Lx = 200.0
Nx = 200
W = Lx / Nx
C11 = 246.5e3
C12 = 147.3e3
C44 = 124.7e3
dLx = 1.0e-10
num_phi = 5
num_grain = 1000
mu = 1.0
kappa = 1.0
L = 1.0
tol = 1.0e-10

comm = MPI.COMM_WORLD

# Step progress and solver residual norm are printed each inc (rank 0 only).

# Only rank 0 prints uncaught exceptions; flush + barrier so the message appears before MPI_Abort
def _mpi_excepthook(etype, value, tb):
    if comm.rank == 0:
        print("\n" + "=" * 60 + "\nError (rank 0 only):", file=sys.stderr)
        print(f"{etype.__name__}: {value}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.__excepthook__(etype, value, tb)
        sys.stdout.flush()
        sys.stderr.flush()
    try:
        comm.barrier()
    except Exception:
        pass
    comm.Abort(1)
sys.excepthook = _mpi_excepthook

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Elastic stiffness tensor (reference orientation) and rotations
# ---------------------------------------------------------------------------
# C0_ijkl for cubic crystal: C12*I_ij*I_kl + C44*(I_ik*I_jl + I_il*I_jk) + (C11-C12-2*C44)*d_ijkl
I = ufl.Identity(3)
i, j, k, l = ufl.indices(4)
dijkl = np.zeros((3, 3, 3, 3))
for a in range(3):
    dijkl[a, a, a, a] = 1.0
dijkl = ufl.as_tensor(dijkl)

C0 = ufl.as_tensor(
    C12 * I[i, j] * I[k, l]
    + C44 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
    + (C11 - C12 - 2 * C44) * dijkl[i, j, k, l],
    (i, j, k, l),
)

C_list = []
for ii in range(num_phi):
    p, q, r, s = ufl.indices(4)
    rot = Rotation.from_euler("zxz", [45 * ii / max(1, num_phi - 1), 0, 0], degrees=True)
    Q = ufl.as_tensor(rot.as_matrix())
    Ctmp = ufl.as_tensor(C0[p, q, r, s] * Q[p, i] * Q[q, j] * Q[r, k] * Q[s, l], (i, j, k, l))
    C_list.append(Ctmp)

# ---------------------------------------------------------------------------
# Mesh and function spaces
# ---------------------------------------------------------------------------
msh = mesh.create_box(
    comm, [[0.0, 0.0, 0.0], [Lx, Lx, W]], [Nx, Nx, 1], cell_type=CellType.hexahedron
)

S = fem.functionspace(msh, ("Lagrange", 1))
V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
T = fem.functionspace(msh, ("Lagrange", 1, (3, 3)))
PF = fem.functionspace(msh, ("Lagrange", 1, (num_phi,)))

# Trial/test for projection (reused for mass matrix and RHS forms)
w_S, s_S = ufl.TrialFunction(S), ufl.TestFunction(S)
w_T, s_T = ufl.TrialFunction(T), ufl.TestFunction(T)

# ---------------------------------------------------------------------------
# Voronoi points for initial grain structure
# ---------------------------------------------------------------------------
points = []
x_sqrt = np.sqrt(1 / num_grain)
for i in range(int(1 / x_sqrt) + 2):
    for j in range(int(1 / x_sqrt) + 2):
        px = np.random.uniform(i * x_sqrt, (i + 1) * x_sqrt)
        py = np.random.uniform(j * x_sqrt, (j + 1) * x_sqrt)
        points.append([px, py])
points = np.array(points)
np.random.shuffle(points)
tree = KDTree(points)


def phi_init_eval(x):
    """Evaluate initial phase-field at points x; x has shape (3, n_points)."""
    n_pts = x.shape[1]
    out = np.zeros((num_phi, n_pts), dtype=np.float64)
    xy = x[0:2, :] / Lx  # (2, n_pts)
    for p in range(n_pts):
        dist, grain_index = tree.query(xy[:, p].reshape(1, -1), k=1)
        dist = dist[0]
        idx = grain_index[0] % num_phi
        out[idx, p] = 1.0 * math.exp(-1.0 * (dist / 0.1) ** 2)
    return out


# ---------------------------------------------------------------------------
# Functions and test/trial functions
# ---------------------------------------------------------------------------
u = fem.Function(V, name="u")
phi = fem.Function(PF, name="phi")
phi_n = fem.Function(PF)
phi.interpolate(phi_init_eval)
phi_n.x.array[:] = phi.x.array

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)
q = ufl.TestFunction(PF)
dq = ufl.TrialFunction(PF)


def eps(uu):
    return ufl.sym(ufl.grad(uu))


def h(phi_vec):
    """Weight vector for stiffness interpolation: h_m / sum(h)."""
    h_phi = []
    h_tot = 0.0
    for m in range(num_phi):
        hm = 0.5 * (1 + ufl.sin(ufl.pi * (phi_vec[m] - 0.5)))
        h_phi.append(hm)
        h_tot = h_tot + hm
    return ufl.as_tensor([h_phi[m] / h_tot for m in range(num_phi)])


def stiffness_and_stress(uu, phi_vec):
    """Effective stiffness C and stress sigma = C : eps(u)."""
    h_phi = h(phi_vec)
    C = h_phi[0] * C_list[0]
    for m in range(1, num_phi):
        C = C + h_phi[m] * C_list[m]
    e = eps(uu)
    # Use fresh indices to avoid UFL "Index out of bounds" (reusing i,j,k,l from C0 conflicts)
    ii, jj, kk, ll = ufl.indices(4)
    sig = ufl.as_tensor(C[ii, jj, kk, ll] * e[kk, ll], (ii, jj))
    return C, sig, e


def free_energy(phi_vec):
    """Double-well + coupling free energy density."""
    f = 0.0
    for m in range(num_phi):
        f = f + mu * (phi_vec[m] ** 4 / 4 - phi_vec[m] ** 2 / 2)
        for n in range(num_phi):
            if n > m:
                f = f + 1.5 * mu * phi_vec[m] ** 2 * phi_vec[n] ** 2
    return f


# ---------------------------------------------------------------------------
# Boundary conditions (displacement)
# ---------------------------------------------------------------------------
def make_bcs(dLx_val):
    facets_x0 = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0, atol=tol)
    )
    facets_x1 = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], Lx, atol=tol)
    )
    facets_y0 = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0, atol=tol)
    )
    facets_z0 = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[2], 0.0, atol=tol)
    )
    facets_origin = mesh.locate_entities_boundary(
        msh,
        msh.topology.dim - 1,
        lambda x: np.isclose(x[0], 0.0, atol=tol)
        & np.isclose(x[1], 0.0, atol=tol)
        & np.isclose(x[2], 0.0, atol=tol),
    )

    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    fdim = msh.topology.dim - 1

    dofs_x0 = fem.locate_dofs_topological(V.sub(0), fdim, facets_x0)
    dofs_x1 = fem.locate_dofs_topological(V.sub(0), fdim, facets_x1)
    dofs_y0 = fem.locate_dofs_topological(V.sub(1), fdim, facets_y0)
    dofs_z0 = fem.locate_dofs_topological(V.sub(2), fdim, facets_z0)
    dofs_oy = fem.locate_dofs_topological(V.sub(1), fdim, facets_origin)
    dofs_oz = fem.locate_dofs_topological(V.sub(2), fdim, facets_origin)

    bc1 = fem.dirichletbc(np.float64(0.0), dofs_x0, V.sub(0))
    bc2 = fem.dirichletbc(np.float64(dLx_val), dofs_x1, V.sub(0))
    bc5 = fem.dirichletbc(np.float64(0.0), dofs_oy, V.sub(1))
    bc6 = fem.dirichletbc(np.float64(0.0), dofs_oz, V.sub(2))
    return [bc1, bc2, bc5, bc6]


# ---------------------------------------------------------------------------
# Projection helper: reuse mass matrix and solver, only reassemble RHS each time
# ---------------------------------------------------------------------------
_proj_scalar_cache = None
_proj_tensor_cache = None


def _init_proj_scalar():
    global _proj_scalar_cache
    if _proj_scalar_cache is not None:
        return
    a_s = ufl.inner(w_S, s_S) * ufl.dx
    form_a_s = fem.form(a_s)
    A_s = create_matrix(form_a_s)
    assemble_matrix(A_s, form_a_s)
    A_s.assemble()
    ksp_s = PETSc.KSP().create(msh.comm)
    ksp_s.setOperators(A_s)
    ksp_s.setType("preonly")
    ksp_s.getPC().setType("lu")
    ksp_s.setUp()
    _proj_scalar_cache = (A_s, ksp_s)


def _init_proj_tensor():
    global _proj_tensor_cache
    if _proj_tensor_cache is not None:
        return
    a_t = ufl.inner(w_T, s_T) * ufl.dx
    form_a_t = fem.form(a_t)
    A_t = create_matrix(form_a_t)
    assemble_matrix(A_t, form_a_t)
    A_t.assemble()
    ksp_t = PETSc.KSP().create(msh.comm)
    ksp_t.setOperators(A_t)
    ksp_t.setType("preonly")
    ksp_t.getPC().setType("lu")
    ksp_t.setUp()
    _proj_tensor_cache = (A_t, ksp_t)


def project_scalar(expr, space_S):
    """Project scalar expression into S; reuses mass matrix and solver."""
    _init_proj_scalar()
    A_s, ksp_s = _proj_scalar_cache
    L = ufl.inner(expr, s_S) * ufl.dx
    form_L = fem.form(L)
    b_s = create_vector(form_L)
    assemble_vector(b_s, form_L)
    b_s.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    result = fem.Function(S)
    ksp_s.solve(b_s, result.x.petsc_vec)
    result.x.scatter_forward()
    return result


def project_tensor(expr, space_T):
    """Project 2nd-order tensor expression into T; reuses mass matrix and solver."""
    _init_proj_tensor()
    A_t, ksp_t = _proj_tensor_cache
    L = ufl.inner(expr, s_T) * ufl.dx
    form_L = fem.form(L)
    b_t = create_vector(form_L)
    assemble_vector(b_t, form_L)
    b_t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    result = fem.Function(T)
    ksp_t.solve(b_t, result.x.petsc_vec)
    result.x.scatter_forward()
    return result


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
dt = 0.1
t = 0.0
inc_max = 502

# Quadrature degree for forms that involve complex expressions
metadata = {"quadrature_degree": 2}

file_phi = io.XDMFFile(comm, out_dir / "phi.xdmf", "w")
file_phi.write_mesh(msh)
file_W = io.XDMFFile(comm, out_dir / "W_elas.xdmf", "w")
file_W.write_mesh(msh)
file_sigma = io.XDMFFile(comm, out_dir / "stress.xdmf", "w")
file_sigma.write_mesh(msh)
file_uij = io.XDMFFile(comm, out_dir / "uij.xdmf", "w")
file_uij.write_mesh(msh)

if comm.rank == 0:
    ncells = msh.topology.index_map(msh.topology.dim).size_global
    print("FEniCSx AGG: setup done, time loop start")
    print(f"  mesh cells (global): {ncells}")
    print(f"  inc_max={inc_max}, dt={dt}, t in [0, {(inc_max-1)*dt:.1f}]")
    print("  dLx = 1e-10 for t<=5, then 1.0")
    print("-" * 60)

for inc in range(inc_max):
    dLx_val = 1.0e-10 if t <= 5.0 else 1.0
    bcs = make_bcs(dLx_val)

    if comm.rank == 0:
        print(f"inc={inc:4d}/{inc_max}  t={t:6.2f}  dLx={dLx_val}", flush=True)

    C, sig, e = stiffness_and_stress(u, phi)

    # Linear elasticity: solve F_el == 0 for u (same as old FEniCS: solve(F == 0, u, bc))
    F_el = ufl.inner(sig, eps(v)) * ufl.dx(metadata=metadata)
    a_el = ufl.derivative(F_el, u, du)

    _el_opts = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-9,
        "snes_max_it": 100,
        "snes_error_if_not_converged": False,
    }
    problem_el = NonlinearProblem(
        F_el,
        u,
        bcs=bcs,
        J=a_el,
        petsc_options_prefix="elastic_",
        petsc_options=_el_opts,
    )
    u = problem_el.solve()
    if comm.rank == 0:
        nel = problem_el.solver.getIterationNumber()
        reas = problem_el.solver.getConvergedReason()
        ok = "ok" if reas > 0 else f"reason={reas}"
        try:
            rnorm = problem_el.solver.getFunctionNorm()
            print(f"  elasticity: {nel} iter, res={rnorm:.2e}, {ok}", flush=True)
        except Exception:
            print(f"  elasticity: {nel} iter, {ok}", flush=True)

    # Recompute strain and stress with solution u
    C, sig, e = stiffness_and_stress(u, phi)
    stress_oldphi = sig  # sigma = C : e already from stiffness_and_stress

    # Phase-field residual: (phi - phi_n)/dt * q + L*dFdphi·q + 0.5*L*dWdphi·q + kappa*L*inner(grad(phi),grad(q))
    # Use derivative of forms w.r.t. phi in direction q
    W_form = 0.5 * ufl.inner(sig, e) * ufl.dx(metadata=metadata)
    f_energy_form = free_energy(phi) * ufl.dx(metadata=metadata)
    F_PF = (
        ufl.inner((phi - phi_n) / dt, q) * ufl.dx(metadata=metadata)
        + L * ufl.derivative(f_energy_form, phi, q)
        + 0.5 * L * ufl.derivative(W_form, phi, q)
        + kappa * L * ufl.inner(ufl.grad(phi), ufl.grad(q)) * ufl.dx(metadata=metadata)
    )
    a_PF = ufl.derivative(F_PF, phi, dq)

    _pf_opts = {
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-9,
        "snes_max_it": 1000,
        "ksp_type": "cg",
        "snes_error_if_not_converged": False,
    }
    if comm.rank == 0 and inc == 0:
        print("  phase-field: JIT-compiling form (first time only, 1–5 min, CPU busy)...", flush=True)
    problem_PF = NonlinearProblem(
        F_PF,
        phi,
        J=a_PF,
        petsc_options_prefix="phase_field_",
        petsc_options=_pf_opts,
    )
    phi = problem_PF.solve()
    if comm.rank == 0:
        npf = problem_PF.solver.getIterationNumber()
        rpf = problem_PF.solver.getConvergedReason()
        ok = "ok" if rpf > 0 else f"reason={rpf}"
        try:
            rnorm = problem_PF.solver.getFunctionNorm()
            print(f"  phase_field: {npf} iter, res={rnorm:.2e}, {ok}", flush=True)
        except Exception:
            print(f"  phase_field: {npf} iter, {ok}", flush=True)
    phi_n.x.array[:] = phi.x.array

    # Output
    out_msg = []
    if inc < 50:
        if inc % 10 < 1e-4:
            file_phi.write_function(phi, np.round(t, 1))
            out_msg.append("phi")
        elif inc % 10 < 1 + 1e-4:
            if comm.rank == 0:
                print("  write: projecting W_elas, stress, grad_u (first time may JIT, 1–3 min)...", flush=True)
            W_elas_expr = 0.5 * ufl.inner(sig, e)
            W_elas = project_scalar(W_elas_expr, S)
            W_elas.name = "W_elas"
            file_W.write_function(W_elas, np.round(t - 0.1, 1))
            grad_u = project_tensor(ufl.grad(u), T)
            grad_u.name = "grad_u"
            file_uij.write_function(grad_u, np.round(t - 0.1, 1))
            stress = project_tensor(stress_oldphi, T)
            stress.name = "stress"
            file_sigma.write_function(stress, np.round(t - 0.1, 1))
            out_msg.append("W_elas,stress,grad_u")
    else:
        if inc % 5 < 1e-4:
            file_phi.write_function(phi, np.round(t, 1))
            out_msg.append("phi")
        if inc % 5 < 1 + 1e-4 and inc % 5 > 1e-4:
            if comm.rank == 0:
                print("  write: projecting W_elas, stress, grad_u...", flush=True)
            W_elas_expr = 0.5 * ufl.inner(sig, e)
            W_elas = project_scalar(W_elas_expr, S)
            W_elas.name = "W_elas"
            file_W.write_function(W_elas, np.round(t - 0.1, 1))
            grad_u = project_tensor(ufl.grad(u), T)
            grad_u.name = "grad_u"
            file_uij.write_function(grad_u, np.round(t - 0.1, 1))
            stress = project_tensor(stress_oldphi, T)
            stress.name = "stress"
            file_sigma.write_function(stress, np.round(t - 0.1, 1))
            out_msg.append("W_elas,stress,grad_u")

    if comm.rank == 0 and out_msg:
        print(f"  write: {', '.join(out_msg)}", flush=True)

    t = t + dt

file_phi.close()
file_W.close()
file_sigma.close()
file_uij.close()

if comm.rank == 0:
    print("-" * 60)
    print("FEniCSx AGG run finished.")
    print("Output in:", out_dir.resolve())
