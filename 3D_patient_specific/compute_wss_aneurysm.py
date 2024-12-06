import argparse
import re

import dolfin as df
import numpy as np
from ufl import block_split

from ns_aneurysm import finite_elements, inflow, mesh_info

comm = df.MPI.comm_world
rank = df.MPI.rank(comm)

desc = "Compute wall shear stress given a HDF5 file of the velocity and pressure (v, p)"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc
)
parser.add_argument(
    "--case",
    default="case01",
    type=str,
    choices=["case01", "case02"],
    dest="case",
    help="which challenge case is it",
)
parser.add_argument(
    "--mesh",
    default="",
    type=str,
    dest="mesh_file",
    help="Path to a marked mesh file (meshname_marked.h5)",
)
parser.add_argument(
    "--element",
    default="th",
    type=str,
    choices=["p1p1", "th"],
    dest="elem",
    help="finite element; p1p1 or th",
)
parser.add_argument(
    "--stab",
    default="none",
    type=str,
    choices=["ip", "none"],
    dest="stab",
    help="stabilization; ip or none",
)
parser.add_argument(
    "-o",
    default="results/",
    type=str,
    dest="res_folder",
    help="results folder",
)
parser.add_argument(
    "--marker-wall",
    default=1,
    type=int,
    dest="mark_wall",
    help="Integer describing which marker is at the walls, i.e the boundary to compute WSS on",
)
parser.add_argument(
    "--refsystems-file",
    default="",
    type=str,
    dest="refsystems_file",
    help="a path to the file with reference systems corresponding to the marked mesh (from vmtk .dat)",
)
parser.add_argument(
    "--mu",
    default=4e-3,
    type=float,
    dest="mu",
    help="Dynamic viscosity in SI units [Pa.s]. Default value is 4 mPa.s",
)
parser.add_argument(
    "--rho",
    default=1000,
    type=float,
    dest="rho",
    help="Density in SI units [kg/m^3]. Default value is 1000 kg/m^3",
)
parser.add_argument(
    "--beta",
    default=100,
    type=float,
    dest="beta",
    help="Parameter for the Nitsche BC. Default value is 100.",
)
parser.add_argument(
    "--w-file",
    default="",
    type=str,
    dest="w_file",
    help="Path to our computed velocity and pressure",
)
parser.add_argument(
    "--compute-differences",
    action="store_true",
    default=False,
    dest="compute_differences",
    help="If true, differences between maximum values of WSS for different FE spaces will be computed.",
)
parser.add_argument(
    "--stationary",
    action="store_true",
    default=False,
    dest="stationary",
    help="Were these results computed for a stationary or a pulsatile flow?",
)
parser.add_argument(
    "--nitsche-noslip",
    action="store_true",
    default=False,
    dest="nitsche_noslip",
    help="if True, Nitsche terms will be added to the form for computing WSS",
)

args = parser.parse_args()
mark_wall = args.mark_wall
mu = args.mu
rho = args.rho
elem = args.elem
stab = args.stab
mesh_file = args.mesh_file
refsystems_file = args.refsystems_file
w_file = args.w_file
stationary = args.stationary

comm = df.MPI.comm_world

# read mesh
mesh = df.Mesh()
with df.HDF5File(comm, mesh_file, "r") as hdf:
    hdf.read(mesh, "/mesh", False)
    dim = mesh.geometry().dim()
    boundary_parts = df.MeshFunction("size_t", mesh, dim - 1, 0)
    hdf.read(boundary_parts, "/boundaries")
mesh.init()


def T(p, v):  # Cauchy stress tensor
    return -p * df.Identity(mesh.geometry().dim()) + 2 * mu * df.sym(df.grad(v))


def tangential_proj(v, n):
    """
    Compute the tangential projection of a vector v given the normal vector n.
    """
    return (df.Identity(v.ufl_shape[0]) - df.outer(n, n)) * v


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
            # self.solver_wss.parameters["absolute_tolerance"] = 1e-12
            # self.solver_wss.parameters["relative_tolerance"] = 1e-10

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


def compute_norm(data, result=None):
    """
    input: quantity living in some VectorElement space (for example velocity or wall shear stress vector)
    output: norm of the quantity living in appropriate FunctionSpace
    """
    V = data.function_space()
    nx_dofs = V.sub(0).dofmap().dofs()
    ny_dofs = V.sub(1).dofmap().dofs()
    nz_dofs = V.sub(2).dofmap().dofs()

    nx = data.vector().vec()[nx_dofs]
    ny = data.vector().vec()[ny_dofs]
    nz = data.vector().vec()[nz_dofs]
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)

    if result is None:
        S = V.sub(0).collapse()  # function space is determined from vector function space V
        result = df.Function(S)
    result.vector().set_local(norm)
    result.vector().apply("insert")
    return result


def pvn(v, n):  # normal component of a vector v
    return df.inner(v, n) * n


def vn(v, n):  # projection of a vector v to the direction of normal vector
    return df.inner(v, n)


def pvt(v, n):  # tangential component of a vector v
    return v - df.inner(v, n) * n


# get finite element spaces
ns_element = getattr(finite_elements, elem)
FE = ns_element(mesh, boundary_parts)
W = FE.W
V = W.sub(0).collapse()
P = W.sub(1).collapse()
dx = FE.dx(metadata={"quadrature_degree": 4})
ds = FE.ds(metadata={"quadrature_degree": 4})
dS = FE.dS(metadata={"quadrature_degree": 4})

# ==================================
# create xdmf files
file_xdmf = dict()
for i in [
    "traction_weak",
    "traction_weak_tangential",
    "wss_weak_DG_0",
    "wss_standard_DG_0",
    "wss_weak_DG_1",
    "wss_standard_DG_1",
    "wss_weak_CG_1",
    "wss_standard_CG_1",
    "wss_weak_DG_0_cp",
    "wss_standard_DG_0_cp",
    "wss_weak_DG_1_cp",
    "wss_standard_DG_1_cp",
    "wss_weak_CG_1_cp",
    "wss_standard_CG_1_cp",
    "v",
]:
    file_xdmf[i] = df.XDMFFile(comm, args.res_folder + i + ".xdmf")
    file_xdmf[i].parameters["flush_output"] = True
    file_xdmf[i].parameters["rewrite_function_mesh"] = False

# Define normal field
n = df.FacetNormal(mesh)

w1 = df.Function(W)
(v, p, v_, p_) = FE.split(w1)
wdot = df.Function(W)
if rank == 0:
    print(f"reading file {w_file}")
fh5 = df.HDF5File(comm, w_file, "r")
ntimesteps = fh5.attributes("/w")["count"]

h = df.Constant(2.0) * df.Circumradius(mesh)


def NitscheBC(eq, n, ds):  # Nitsche's method implemented through the derivative of a functional
    w_ = df.TestFunction(w1.function_space())
    penalty = (args.beta * mu / h) * df.inner(eq, df.derivative(eq, w1, w_)) * ds
    bcpart = (
        df.inner(T(p, v) * n, df.derivative(eq, w1, w_)) * ds
        - df.inner(df.derivative(T(p, v) * n, w1, w_), eq) * ds
    )
    return -bcpart + penalty


if refsystems_file == "":
    refsystems_file = f"meshes/{args.case}_refsystems_SI.dat"
refsys = mesh_info.read_file(refsystems_file)  # output from vmtk

# identify inflow id as the end with largest diameter
rr = 0.0
inflow_idx = 0
for i in range(len(refsys)):
    if refsys[i].r > rr:
        rr = refsys[i].r
        inflow_idx = i

# select the remaining ids as outflow
outflow_idx = []
for i in range(len(refsys)):
    if (i != inflow_idx) and (refsys[i].r > 0.0):
        outflow_idx.append(i)

# mark boundaries
mark_in = 2  # since 0 is mark for interior facets and 1 for walls
marks_out = list(range(3, len(refsys) + 2))

mesh_in_out = []
j = 0
for i in range(len(refsys)):
    nn = (refsys[i].nx, refsys[i].ny, refsys[i].nz)
    r = refsys[i].r
    s = (refsys[i].sx, refsys[i].sy, refsys[i].sz)
    if i == inflow_idx:
        mark = mark_in
    else:
        mark = marks_out[j]
        j = j + 1
    mesh_in_out.append(mesh_info.MeshInOut(n=nn, s=s, r=r, mark=mark))


# ==================================
# assemble the LHS

a = df.Function(W)
(_, _, v_test, p_test) = FE.split(a)

# ensure P1 recovery for TH element
V = df.VectorFunctionSpace(mesh, "P", 1)

g = df.TrialFunction(V)
g_test = df.TestFunction(V)

lhs = df.inner(g, g_test) * ds(mark_wall)

A = df.assemble(lhs, keep_diagonal=True)
A.ident_zeros()

solver_wss = df.KrylovSolver("bicgstab", "jacobi")
solver_wss.set_operator(A)


def evaluate_wss_timestep(i: int, append: bool = False):
    # Compute WSS weakly

    fh5.read(w1, f"/w/vector_{i}")
    fh5.read(wdot, f"/wdot/vector_{i}")
    t = fh5.attributes(f"/w/vector_{i}").to_dict().get("timestamp")
    v = w1.sub(0, deepcopy=True)
    p = w1.sub(1, deepcopy=True)
    vdot = wdot.sub(0, deepcopy=True)

    # weak form
    F_wss = (
        rho * df.inner(vdot, v_test) * dx
        + rho * df.inner(df.grad(v) * v, v_test) * dx
        + df.inner(T(p, v), df.grad(v_test)) * dx
        + df.div(v) * p_test * dx
        - df.inner(T(p, v) * n, v_test) * ds(mark_in)
    )

    # add Neumann BC on outlets (not necessary)
    for j in marks_out:
        F_wss += -df.inner(T(p, v) * n, v_test) * ds(j)

    # add nitsche terms
    if args.nitsche_noslip:
        F_wss += df.inner(T(p_test, v_test) * n, v - df.Constant((0.0, 0.0, 0.0))) * ds(
            mark_wall
        )

        F_wss += (
            (args.beta * mu / h)
            * df.inner(v - df.Constant((0.0, 0.0, 0.0)), v_test)
            * ds(mark_wall)
        )

        v_in = inflow.InflowAnalyticalStac(
            mesh_in_out[inflow_idx].s,
            mesh_in_out[inflow_idx].r,
            mesh_in_out[inflow_idx].n,
            v_mean=0.5,
        )

        # empirical term (compensation for Nitsche BC on the inlet)
        F_wss -= NitscheBC(v - v_in, n, ds(mark_in))

    if stab == "ip":
        # add interior penalty stabilization
        # eq. 15 in https://link.springer.com/content/pdf/10.1007/s00211-007-0070-5.pdf
        # as we are not using dimensionless equations, we added the density
        alpha_i = 1e-3 * rho
        # alpha_v = 1e-2 * rho
        alpha_v = 1e-3 * rho
        alpha_p = 1.0 / rho
        F_stab = (
            alpha_i
            * df.avg(h) ** 2
            * pow(df.dot(v("+"), n("+")), 2)
            * df.inner(df.jump(df.grad(v)), df.jump(df.grad(v_test)))
            * dS
        )
        F_stab += (
            alpha_v
            * df.avg(h) ** 2
            * df.inner(df.jump(df.grad(v)), df.jump(df.grad(v_test)))
            * dS
        )
        F_stab += (
            alpha_p
            * df.avg(h) ** 2
            * df.inner(df.jump(df.grad(p)), df.jump(df.grad(p_test)))
            * dS
        )
        F_wss += F_stab

    # compute full traction vector weakly (it is actually redundant)
    WFv = block_split(F_wss, 0)
    rhs = df.assemble(df.action(WFv, g_test))
    traction_weak = df.Function(V, name="traction_force")
    solver_wss.solve(traction_weak.vector(), rhs)
    file_xdmf["traction_weak"].write_checkpoint(
        traction_weak, "traction_force", t, append=append
    )

    # compute the tangential part of traction force weakly
    F_wss_tan = F_wss - (df.inner(df.inner(T(p, v) * n, n) * n, v_test) * ds(mark_wall))

    WFv = block_split(F_wss_tan, 0)
    rhs = df.assemble(df.action(WFv, g_test))
    traction_weak_tan = df.Function(V, name="wss_weak")
    solver_wss.solve(traction_weak_tan.vector(), rhs)
    file_xdmf["traction_weak_tangential"].write_checkpoint(
        traction_weak_tan, "wss", t, append=append
    )
    file_xdmf["v"].write_checkpoint(v, "velocity", t, append=append)
    # v.rename("v", "velocity")
    # file_xdmf["v"].write(v, t)

    # ==================================
    # Project weak traction onto the desired function space

    wss_spaces = [("DG", 0), ("DG", 1), ("CG", 1)]

    for wss_family, wss_degree in wss_spaces:

        # print(f"{wss_family} {wss_degree}")
        traction_element = df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree)
        traction_space = df.FunctionSpace(mesh, traction_element)
        traction_computer = TractionComputation(traction_space, mark_wall=mark_wall)

        wss_weak = traction_computer.project_function(traction_weak_tan)

        # standard evaluation
        Tn = T(p, v) * n
        wss = traction_computer.project_wss(Tn, n)

        file_xdmf[f"wss_weak_{wss_family}_{wss_degree}_cp"].write_checkpoint(
            wss_weak, "wss", t, append=append
        )
        file_xdmf[f"wss_standard_{wss_family}_{wss_degree}_cp"].write_checkpoint(
            wss, "wss", t, append=append
        )
        wss_weak.rename("wss", "wss")
        file_xdmf[f"wss_weak_{wss_family}_{wss_degree}"].write(wss_weak, t)
        wss.rename("wss", "wss")
        file_xdmf[f"wss_standard_{wss_family}_{wss_degree}"].write(wss, t)

    if args.compute_differences:

        if stationary:
            i = 0
        # DG0
        wss_weak_DG0 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("DG", mesh.ufl_cell(), 0))
        )
        file_xdmf["wss_weak_DG_0_cp"].read_checkpoint(wss_weak_DG0, "wss", i)
        wss_weak_DG0_norm = compute_norm(wss_weak_DG0)
        wss_weak_DG0_max = wss_weak_DG0_norm.vector().max()

        # DG1
        wss_weak_DG1 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("DG", mesh.ufl_cell(), 1))
        )
        file_xdmf["wss_weak_DG_1_cp"].read_checkpoint(wss_weak_DG1, "wss", i)
        wss_weak_DG1_norm = compute_norm(wss_weak_DG1)
        wss_weak_DG1_max = wss_weak_DG1_norm.vector().max()

        # CG1
        wss_weak_CG1 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("CG", mesh.ufl_cell(), 1))
        )
        file_xdmf["wss_weak_CG_1_cp"].read_checkpoint(wss_weak_CG1, "wss", i)
        wss_weak_CG1_norm = compute_norm(wss_weak_CG1)
        wss_weak_CG1_max = wss_weak_CG1_norm.vector().max()

        # relative percentage differences
        diff_weak_DG0 = (wss_weak_DG0_max - wss_weak_CG1_max) / wss_weak_CG1_max * 100
        diff_weak_DG1 = (wss_weak_DG1_max - wss_weak_CG1_max) / wss_weak_CG1_max * 100

        # DG0
        wss_DG0 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("DG", mesh.ufl_cell(), 0))
        )
        file_xdmf["wss_standard_DG_0_cp"].read_checkpoint(wss_DG0, "wss", i)
        wss_DG0_norm = compute_norm(wss_DG0)
        wss_DG0_max = wss_DG0_norm.vector().max()

        # DG1
        wss_DG1 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("DG", mesh.ufl_cell(), 1))
        )
        file_xdmf["wss_standard_DG_1_cp"].read_checkpoint(wss_DG1, "wss", i)
        wss_DG1_norm = compute_norm(wss_DG1)
        wss_DG1_max = wss_DG1_norm.vector().max()

        # CG1
        wss_CG1 = df.Function(
            df.FunctionSpace(mesh, df.VectorElement("CG", mesh.ufl_cell(), 1))
        )
        file_xdmf["wss_standard_CG_1_cp"].read_checkpoint(wss_CG1, "wss", i)
        wss_CG1_norm = compute_norm(wss_CG1)
        wss_CG1_max = wss_CG1_norm.vector().max()

        # relative percentage differences
        diff_DG0 = (wss_DG0_max - wss_CG1_max) / wss_CG1_max * 100
        diff_DG1 = (wss_DG1_max - wss_CG1_max) / wss_CG1_max * 100

        if rank == 0:
            print(f"{wss_weak_DG0_max=}")
            print(f"{wss_weak_DG1_max=}")
            print(f"{wss_weak_CG1_max=}")
            print(f"{wss_DG0_max=}")
            print(f"{wss_DG1_max=}")
            print(f"{wss_CG1_max=}")
            print(f"{diff_weak_DG0=}")
            print(f"{diff_weak_DG1=}")
            print(f"{diff_DG0=}")
            print(f"{diff_DG1=}")
            print("\n")


if stationary:
    evaluate_wss_timestep(ntimesteps - 1, append=False)
else:
    evaluate_wss_timestep(0, append=False)
    for i in range(1, ntimesteps):
        evaluate_wss_timestep(i, append=True)
