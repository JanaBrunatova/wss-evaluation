import os
import sys

import mpi4py
import petsc4py

petsc4py.init(sys.argv)

import dolfin as df
import numpy as np
from ufl import block_split

from ns_aneurysm import finite_elements
from ns_aneurysm.ns_parameters import (
    Theta,
    boundary_parts,
    dest,
    element,
    marks,
    mesh,
    meshname,
    mu,
    rho,
    v_avg,
)

mpi_py = mpi4py.MPI
PETSc = petsc4py.PETSc
mpi_start_time = mpi_py.Wtime()
print = PETSc.Sys.Print

comm = df.MPI.comm_world
rank = df.MPI.rank(comm)

df.parameters["std_out_all_processes"] = False
df.parameters["allow_extrapolation"] = False
df.parameters["ghost_mode"] = "shared_facet"  #  "shared_facet", "shared_vertex", "none"
df.parameters["form_compiler"]["quadrature_degree"] = 7
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_opts = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
    "quadrature_degree": 4,
    "cpp_optimize": True,
}

ns_element = getattr(finite_elements, element)
FE = ns_element(mesh, boundary_parts)

W = FE.W
dx = FE.dx(metadata={"quadrature_degree": 7})
ds = FE.ds(metadata={"quadrature_degree": 7})
dS = FE.dS(metadata={"quadrature_degree": 7})

print(f"{element=!s}")
print(f"{W.dim()=}")

facet_normal = df.FacetNormal(mesh)

# define the radius and length of the cylinder
R = 0.001  # m
L = 0.002  # m
v_max = 2 * v_avg
nu = mu / rho
visc = nu  # can be manually switched to mu if you wish (with proper scaling of gammas and beta)
# visc = mu
delta_p = 4 * visc * v_max * L / R**2  # pressure drop

I = df.Identity(mesh.geometry().dim())  # Identity tensor

h = df.Constant(2.0) * df.Circumradius(mesh)

s = df.conditional(df.gt(nu, df.avg(h)), df.Constant(2.0), df.Constant(1.0))

# define weights for stabilization and Nitsche BC
gamma_v = 1e-3
gamma_p = 1.0
beta = 10.0 * nu
# scaling weights by density if necessary
if visc == mu:
    gamma_v = gamma_v * rho
    gamma_p = gamma_p / rho
    beta = beta * rho


def T(p, v):  # Cauchy stress tensor
    return -p * I + 2 * visc * df.sym(df.grad(v))


def tangential_proj(v, n):
    """
    Compute the tangential projection of a vector v given the normal vector n.
    """
    return (df.Identity(v.ufl_shape[0]) - df.outer(n, n)) * v


# exact formulas for velocity and pressure
u_profile = "2.0*v_avg*(pow(r,2)-(pow(x[0],2)+pow(x[1],2)))/(r*r)"
u_exact = df.Expression(("0.0", "0.0", u_profile), r=R, v_avg=v_avg, degree=2)
p_exact = df.Expression("P*(1-x[2]/L)", P=delta_p, L=L, degree=2)


# Set the folder where solutions will be saved
folder = os.path.join(dest, meshname, element)
print(folder)

# Define unknowns and test function(s)
w = df.Function(W)
v_, p_ = df.TestFunctions(W)
v, p = df.TrialFunctions(W)

v, p = df.split(w)

# weak formulation
F = df.inner(T(p, v), df.grad(v_)) * dx + df.div(v) * p_ * dx


def NitscheBC(
    eq, n, ds, beta=1e3
):  # Nitsche's method implemented through the derivative of a functional
    w_ = df.TestFunction(w.function_space())
    penalty = (beta / h) * df.inner(eq, df.derivative(eq, w, w_)) * ds
    bcpart = (
        df.inner(T(p, v) * n, df.derivative(eq, w, w_)) * ds
        - df.inner(df.derivative(T(p, v) * n, w, w_), eq) * ds
    )
    return -bcpart + penalty


if Theta == -1.0:
    bc_name = "Dirichlet_noslip"
    # Boundary conditions
    bcs = []
    # No-slip boundary condition for velocity on walls
    bc0 = df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), boundary_parts, marks["wall"])
    bcin = df.DirichletBC(W.sub(0), u_exact, boundary_parts, marks["in"])
    bcs = [bcin, bc0]
elif Theta == 1.0:
    bc_name = "Nitsche_noslip"
    # Nitsche wall BC
    Fbc = NitscheBC(v - u_exact, facet_normal, ds(marks["wall"]), beta=beta)
    # Nitsche inflow BC
    Fbc += NitscheBC(v - u_exact, facet_normal, ds(marks["in"]), beta=beta)
    F += Fbc
    bcs = []
else:
    raise NotImplementedError("TODO: fix for theta < 1.0")

bcout_x = df.DirichletBC(W.sub(0).sub(0), df.Constant(0.0), boundary_parts, marks["out"][0])
bcs.append(bcout_x)
bcout_y = df.DirichletBC(W.sub(0).sub(1), df.Constant(0.0), boundary_parts, marks["out"][0])
bcs.append(bcout_y)


def j(p, q):
    return (
        0.5
        * gamma_p
        * df.avg(h) ** (1 + s)
        * df.inner(df.jump(df.grad(p), facet_normal), df.jump(df.grad(q), facet_normal))
        * dS
    )


def j_tilde(u, v):
    return (
        0.5
        * gamma_v
        * df.avg(h) ** (1 + s)
        * df.inner(df.jump(df.div(u)), df.jump(df.div(v)))
        * dS
    )


if element == "p1p1":

    F_stab = j_tilde(v, v_) + j(p, p_)
    F += F_stab


problem = df.NonlinearVariationalProblem(F, w, bcs, J=df.derivative(F, w))
solver = df.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-14
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-14
solver.solve()

v, p = w.split(deepcopy=True)
v.rename("v", "velocity")
p.rename("p", "pressure")

resNS_file_xdmf = os.path.join(folder, bc_name, "%s.xdmf")
resNS_file_hdf5 = os.path.join(folder, bc_name, "%s.h5")

file_xdmf = dict()
file_hdf5 = dict()

for i in ["v", "p", "weak_traction", "weak_tangential_traction"]:
    file_xdmf[i] = df.XDMFFile(comm, resNS_file_xdmf % i)
    file_xdmf[i].parameters["flush_output"] = True
    file_xdmf[i].parameters["rewrite_function_mesh"] = False
file_hdf5["w"] = df.HDF5File(comm, resNS_file_hdf5 % "w", "w")

file_xdmf["v"].write(v)
file_xdmf["p"].write(p)

# compute L2 errors for velocity and pressure
L2_error_velocity = np.sqrt(df.assemble(df.dot(v - u_exact, v - u_exact) * dx))
L2_error_pressure = np.sqrt(df.assemble(df.dot(p - p_exact, p - p_exact) * dx))
print(f"{L2_error_velocity=}")
print(f"{L2_error_pressure=}")
with open(os.path.join(folder, bc_name, "L2_errors.txt"), "w") as f:
    f.write("L2 norm of the velocity and pressure wrt analytical solution\n")
    f.write(str(L2_error_velocity))
    f.write("\n")
    f.write(str(L2_error_pressure))

file_hdf5["w"].write(w, "w", 0)
file_hdf5["w"].close()


### COMPUTE TRACTION AND WALL SHEAR STRESSES ###


class TractionComputation:
    def __init__(
        self,
        traction_space,
        mark_wall: int = 1,
        direct_solver: bool = False,
    ):
        self.mark_wall = mark_wall
        self.space = traction_space

        # Set quantities to compute the WSS
        x = df.TrialFunction(self.space)
        self.x_ = df.TestFunction(self.space)
        lhs = df.inner(x, self.x_) * ds(mark_wall)

        A = df.assemble(lhs, keep_diagonal=True)
        A.ident_zeros()

        if direct_solver:
            self.solver_wss = df.LUSolver("mumps")
        else:
            self.solver_wss = df.KrylovSolver("bicgstab", "jacobi")
            self.solver_wss.parameters["absolute_tolerance"] = 1e-14
            self.solver_wss.parameters["relative_tolerance"] = 1e-14

        self.solver_wss.set_operator(A)

    def project_function(self, b: df.Function) -> df.Function:
        """
        Compute traction force as a projection of a function b.
        """
        traction = df.Function(self.space)
        Ln = df.inner(b, self.x_) * ds(self.mark_wall)
        rhs = df.assemble(Ln)
        self.solver_wss.solve(traction.vector(), rhs)
        return traction

    def project_wss(self, b, n):
        """
        Compute wall shear stress as a projection of a function b.
        """
        wss = df.Function(self.space)
        b_t = tangential_proj(b, n)
        Ln = df.inner(b_t, self.x_) * ds(self.mark_wall)
        rhs = df.assemble(Ln)
        self.solver_wss.solve(wss.vector(), rhs)
        return wss


WW = FE.W
V = df.VectorFunctionSpace(mesh, "P", 1)
a = df.Function(WW)
(_, _, v_test, p_test) = FE.split(a)
g = df.TrialFunction(V)
g_test = df.TestFunction(V)

lhs = df.inner(g, g_test) * ds(marks["wall"]) + df.Constant(0) * df.inner(g, g_test) * df.dx

A = df.assemble(lhs, keep_diagonal=True)
A.ident_zeros()

F_wss = (
    df.inner(T(p, v), df.grad(v_test)) * dx
    + df.div(v) * p_test * dx
    - (df.inner(T(p, v) * facet_normal, v_test) * ds(marks["in"]))
    - (df.inner(T(p, v) * facet_normal, v_test) * ds(marks.get("out")[0]))
)
if element == "p1p1":
    F_wss += F_stab

if Theta == 1.0:
    F_wss += df.inner(T(p_test, v_test) * facet_normal, v - u_exact) * ds(marks["wall"])
    F_wss += (beta / h) * df.inner(v - u_exact, v_test) * ds(marks["wall"])

    F_wss += NitscheBC(v - u_exact, facet_normal, ds(marks["in"]), beta=beta)

    error_noslip = np.sqrt(df.assemble(df.dot(v - u_exact, v - u_exact) * ds(marks["wall"])))
    print(f"{error_noslip=}")

WFv = block_split(F_wss, 0)
rhs = df.assemble(df.action(WFv, g_test))

traction_weak = df.Function(V, name="traction_force")

solver = df.LUSolver("mumps")
solver.set_operator(A)
solver.solve(traction_weak.vector(), rhs)
file_xdmf["weak_traction"].write_checkpoint(traction_weak, "traction_force", append=False)

# analytical expression for traction
traction_exact = df.Expression(
    (
        "-P*(1-x[2]/L)*cos(atan2(x[1],x[0]))",
        "-P*(1-x[2]/L)*sin(atan2(x[1],x[0]))",
        "-(P*R)/(2*L)",
    ),
    P=delta_p,
    R=R,
    L=L,
    degree=2,
)

error_traction = np.sqrt(
    df.assemble(
        df.dot(
            (traction_weak - traction_exact),
            (traction_weak - traction_exact),
        )
        * ds(marks["wall"])
    )
)

print(f"{error_traction=}")

F_wss_tan = F_wss - (
    df.inner(
        df.inner(T(p, v) * facet_normal, facet_normal) * facet_normal,
        v_test,
    )
    * ds(marks["wall"])
)

WFv = block_split(F_wss_tan, 0)
rhs = df.assemble(df.action(WFv, g_test))

traction_weak_tan = df.Function(V, name="wss_weak")

solver = df.LUSolver("mumps")
solver.set_operator(A)
solver.solve(traction_weak.vector(), rhs)
solver.solve(traction_weak_tan.vector(), rhs)

file_xdmf["weak_tangential_traction"].write_checkpoint(
    traction_weak_tan, "wss_weak", append=False
)

# analytical expression for wss
wss_exact = df.Expression(
    ("0", "0", "-(P*R)/(2*L)"),
    P=delta_p,
    R=R,
    L=L,
    degree=0,
)
error_traction_tan = np.sqrt(
    df.assemble(
        df.dot(
            (traction_weak_tan - wss_exact),
            (traction_weak_tan - wss_exact),
        )
        * ds(marks["wall"])
    )
)

print(f"{error_traction_tan=}")


def save_wss(wss_family, wss_degree):

    traction_element = df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree)
    traction_space = df.FunctionSpace(mesh, traction_element)
    traction_computer = TractionComputation(
        traction_space, mark_wall=marks["wall"], direct_solver=True
    )

    for i in [
        f"wss_weak_{wss_family}_{wss_degree}",
        f"wss_standard_{wss_family}_{wss_degree}",
        f"traction_standard_{wss_family}_{wss_degree}",
    ]:
        file_xdmf[i] = df.XDMFFile(comm, resNS_file_xdmf % i)
        file_xdmf[i].parameters["flush_output"] = True
        file_xdmf[i].parameters["rewrite_function_mesh"] = False

    # weak evaluation
    # # project the tangential part of traction we computed before
    wss_weak = traction_computer.project_function(traction_weak_tan)
    error_wss_weak = np.sqrt(
        df.assemble(df.dot((wss_weak - wss_exact), (wss_weak - wss_exact)) * ds(marks["wall"]))
    )

    # standard evaluation
    Tn = T(p, v) * facet_normal
    standard_projection = traction_computer.project_function(Tn)
    file_xdmf[f"traction_standard_{wss_family}_{wss_degree}"].write_checkpoint(
        standard_projection, "traction_force", append=False
    )
    error_standard_traction = np.sqrt(
        df.assemble(
            df.dot(
                (standard_projection - traction_exact),
                (standard_projection - traction_exact),
            )
            * ds(marks["wall"])
        )
    )
    wss = traction_computer.project_wss(Tn, facet_normal)
    error_wss = np.sqrt(
        df.assemble(df.dot((wss - wss_exact), (wss - wss_exact)) * ds(marks["wall"]))
    )

    print(f"{wss_family=} {wss_degree=}")
    print(f"{error_standard_traction=}")
    print(f"{error_wss_weak=}")
    print(f"{error_wss=}")

    wss_weak.rename("wss", "wss")
    file_xdmf[f"wss_weak_{wss_family}_{wss_degree}"].write_checkpoint(wss_weak, "wss", 0)
    file_xdmf[f"wss_standard_{wss_family}_{wss_degree}"].write_checkpoint(wss, "wss", 0)


wss_spaces = [("DG", 0), ("CG", 1), ("DG", 1)]

for wss_family, wss_degree in wss_spaces:
    save_wss(wss_family, wss_degree)
