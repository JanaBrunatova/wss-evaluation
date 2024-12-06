import os
import pprint
import shutil
import sys

import mpi4py
import petsc4py

petsc4py.init(sys.argv)

import dolfin as df
import numpy as np
from ufl import block_split

import ns_aneurysm.petsc_ts_solver as TS
import ns_aneurysm.solver_settings as solver_settings
from ns_aneurysm import finite_elements, generate_normal, model, stabilizations
from ns_aneurysm.ns_parameters import (
    Theta,
    Theta_in,
    all_monitor,
    basic_monitor,
    bcout_dir_do_nothing,
    bcout_poiseuille,
    beta,
    boundary_parts,
    compare_with_poiseuille,
    dest,
    dt,
    element,
    gamma,
    marks,
    mesh,
    meshname,
    model_name_str,
    mu,
    nitsche_type,
    normal,
    opts,
    periods,
    profile,
    restart,
    restart_stepsback,
    restart_t_begin,
    rho,
    save_bndry_marks,
    scaling_factor,
    stab,
    t_atol,
    t_begin,
    t_end,
    t_period,
    t_rtol,
    ts_adapt_type,
    uniform_dt,
    uniform_dt_last_period,
    unit_system,
    v_avg,
    v_in,
    xdmf_all,
    xdmf_last,
)

mpi_py = mpi4py.MPI
PETSc = petsc4py.PETSc
mpi_start_time = mpi_py.Wtime()
print = PETSc.Sys.Print

comm = df.MPI.comm_world
rank = df.MPI.rank(comm)

df.parameters["std_out_all_processes"] = False
df.parameters["allow_extrapolation"] = False
df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["quadrature_degree"] = 4
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
dx = FE.dx(metadata={"quadrature_degree": 4})
ds = FE.ds(metadata={"quadrature_degree": 4})
dS = FE.dS(metadata={"quadrature_degree": 4})
print(W.dim())

# Define normal fields
n = dict()
for i in ["wall", "in"]:
    if normal == "FacetNormal":
        n[marks[i]] = df.FacetNormal(mesh)
    elif normal == "proj":
        n[marks[i]] = generate_normal.make_normal_projection(
            mesh, boundary_parts, id=marks[i], type="CG1"
        )
    else:
        print("Wrong normal type.")

for i in marks.get("out"):
    if normal == "FacetNormal":
        n[i] = df.FacetNormal(mesh)
    elif normal == "proj":
        n[i] = generate_normal.make_normal_projection(mesh, boundary_parts, id=i, type="CG1")

# Define unknown and test function(s)
w = df.Function(W)
(v, p, v_, p_) = FE.split(w)
wdot = df.Function(W)
(vdot, pdot) = FE.split(wdot, test=False)

I = df.Identity(mesh.geometry().dim())  # Identity tensor
# edgelen = mesh.hmin()  # minimum cell diameter
# edgelen = df.MinCellEdgeLength(mesh)  # minimum cell diameter
edgelen = df.Constant(2.0) * df.Circumradius(mesh)

# save boundary marks
if save_bndry_marks:
    if rank == 0:
        if not os.path.exists("mesh_labels/"):
            os.makedirs("mesh_labels/")
    if scaling_factor != 1.0:
        xdmf = df.XDMFFile(
            comm,
            "mesh_labels/boundary_parts_"
            + meshname
            + "_scale_"
            + str(scaling_factor)
            + ".xdmf",
        )
    else:
        xdmf = df.XDMFFile(comm, "mesh_labels/boundary_parts_" + meshname + ".xdmf")
    xdmf.write(boundary_parts)
    xdmf.close()

# Set viscosity model from model.py
model_name = getattr(model, model_name_str)
scale_CGS_to_SI = 0.1  # Poise to SI

if unit_system == "SI":
    if model_name_str == "Newtonian":
        visc_model = model_name(mesh, v, scale=1.0, mu=mu)
    else:
        visc_model = model_name(mesh, v, scale=scale_CGS_to_SI)
elif model_name_str == "Newtonian":  # unit_system=='CGS'
    visc_model = model_name(mesh, v, scale=1.0, mu=mu)
    # the scale has to be 1 because mu has been already scaled in ns_parameters.py
else:
    visc_model = model_name(mesh, v, scale=1.0)


def T(p, v):  # Cauchy stress tensor
    return -p * I + 2.0 * visc_model.viscosity * df.sym(df.grad(v))
    # return -p * I + visc_model.viscosity * df.grad(v)


def pvn(v, n):  # normal component of a vector v
    return df.inner(v, n) * n


def vn(v, n):  # projection of a vector v to the direction of normal vector
    return df.inner(v, n)


def pvt(v, n):  # tangential component of a vector v
    return v - df.inner(v, n) * n


def NitscheBC(eq, n, ds):  # Nitsche's method implemented through the derivative of a functional
    w_ = df.TestFunction(w.function_space())
    penalty = (beta * mu / edgelen) * df.inner(eq, df.derivative(eq, w, w_)) * ds
    # penalty = (beta * mu / edgelen) * df.derivative(df.inner(eq, eq), w, w_) * ds
    if nitsche_type == "sym":
        bcpart = (
            df.inner(T(p, v) * n, df.derivative(eq, w, w_)) * ds
            + df.inner(df.derivative(T(p, v) * n, w, w_), eq) * ds
        )
    elif nitsche_type == "nonsym":
        bcpart = (
            df.inner(T(p, v) * n, df.derivative(eq, w, w_)) * ds
            - df.inner(df.derivative(T(p, v) * n, w, w_), eq) * ds
        )
    else:
        raise ValueError("Invalid nitsche_type value.")
    return -bcpart + penalty


# weak form
F = (
    rho * df.inner(vdot, v_) * dx
    + rho * df.inner(df.grad(v) * v, v_) * dx
    + df.inner(T(p, v), df.grad(v_)) * dx
    + df.div(v) * p_ * dx
)

# strong form
eqv = rho * vdot + rho * df.grad(v) * v - df.div(T(p, v))
eqp = df.div(v)

# Boundary conditions
bcs = []
if Theta == -1.0:
    # No-slip boundary condition for velocity on walls
    bc0 = df.DirichletBC(W.sub(0), df.Constant((0.0, 0.0, 0.0)), boundary_parts, marks["wall"])
    # Inflow velocity
    bcin = df.DirichletBC(W.sub(0), v_in, boundary_parts, marks["in"])
    # Outflow condition
    if bcout_poiseuille:
        bcout_x = df.DirichletBC(
            W.sub(0).sub(0), df.Constant(0.0), boundary_parts, marks["out"][0]
        )
        bcout_y = df.DirichletBC(
            W.sub(0).sub(1), df.Constant(0.0), boundary_parts, marks["out"][0]
        )
        bcs = [bcin, bcout_x, bcout_y, bc0]
    else:
        bcs = [bcin, bc0]
    bc_name = "Dirichlet_noslip"
else:
    bcin = df.DirichletBC(W.sub(0), v_in, boundary_parts, marks["in"])
    bcs = []
    if Theta_in == -1.0:
        bcs.append(bcin)
    if Theta == 1.0:
        bc_name = "Nitsche_noslip"
    elif Theta == 0.999:
        bc_name = "Nitsche_Navier_slip_{0:03d}".format(int(1000 * Theta))
    else:
        bc_name = "Nitsche_Navier_slip_{0:02d}".format(int(100 * Theta))

# Outflow directional do-nothing condition
if bcout_dir_do_nothing:
    for j in marks.get("out"):
        F += (
            -0.5
            * rho
            * df.inner(df.conditional(df.gt(vn(v, n[j]), 0.0), 0.0, 1.0) * vn(v, n[j]) * v, v_)
            * ds(j)
        )

if Theta > -1.0:
    # Nitsche - zero normal on the wall
    if Theta == 1.0:
        Fbc = NitscheBC(v - df.Constant((0.0, 0.0, 0.0)), n[marks["wall"]], ds(marks["wall"]))
    else:
        Fbc = NitscheBC(pvn(v, n[marks["wall"]]), n[marks["wall"]], ds(marks["wall"]))
    if Theta < 1.0:
        # Navier tangential slip on walls as Neumann
        Fbc += df.inner(
            (Theta / (gamma * (1.0 - Theta))) * pvt(v, n[marks["wall"]]),
            pvt(v_, n[marks["wall"]]),
        ) * ds(marks["wall"])
    # else:
    #     Fbc += NitscheBC(pvt(v, n[marks["wall"]]), n[marks["wall"]], ds(marks["wall"]))
    # Nitsche - inflow
    if Theta_in > -1.0:
        if Theta_in < 1.0:
            Fbc += NitscheBC(pvn(v - v_in, n[marks["in"]]), n[marks["in"]], ds(marks["in"]))
            Fbc += NitscheBC(
                Theta_in * pvt(v - v_in, n[marks["in"]]), n[marks["in"]], ds(marks["in"])
            )
        else:
            Fbc += NitscheBC(v - v_in, n[marks["in"]], ds(marks["in"]))
    F += Fbc

# Set the folder where solutions will be saved
if profile != "pulsatile":
    folder = os.path.join(
        dest,
        meshname,
        "stationary",
        normal + "_" + element + "_" + stab,
        bc_name,
        visc_model.__class__.__name__,
    )
else:
    folder = os.path.join(
        dest,
        meshname,
        "pulsatile",
        normal + "_" + element + "_" + stab,
        bc_name,
        visc_model.__class__.__name__,
    )


# Possibly add stabilization
stabilization = getattr(stabilizations, stab)

ST = stabilization(
    mesh,
    boundary_parts,
    FE,
    (v, p, vdot, pdot),
    (eqv, eqp),
    (v_, p_),
    rho=rho,
    viscosity=visc_model.viscosity,
    k=1.0 / dt,
)
F += ST.S + FE.stab(w)

J = df.derivative(F, w)
Jwdot = df.derivative(F, wdot)


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


# update - is called before every assembly, one can update timedependent bc/rhs
def update(ts, t):
    ctx = ts.getAppCtx()
    inflow = ctx.get("inflow")
    if inflow is not None:
        inflow.update(t)
    outflow = ctx.get("outflow")
    problem = ctx.get("problem")
    if problem.stab is not None:
        problem.stab.update(ts)
    visc_model.project_viscosity()


# prestep is called at the start of every timestep
def prestep(ts):
    t = ts.getTime()
    it = ts.step_number
    ctx = ts.getAppCtx()
    problem = ctx.get("problem")
    solver = ctx.get("solver")
    nits = ts.snes.getIterationNumber()


# report - is called after every succesfull timestep, one can save xdmf there
def report(ts):
    t = ts.getTime()
    dt = ts.getTimeStep()
    cputime = mpi_py.Wtime() - mpi_start_time

    ctx = ts.getAppCtx()
    problem = ctx.get("problem")
    solver = ctx.get("solver")
    outflow = ctx.get("outflow")
    FE = ctx.get("FE")

    if ts.iterating or ts.converged:
        (v, p) = FE.extract(w)
        (vdot, pdot) = FE.extract(wdot)
        fin = df.assemble(-df.inner(v, n[marks["in"]]) * ds(marks["in"]))
        fwall = df.assemble(df.inner(v, n[marks["wall"]]) * ds(marks["wall"]))
        fout = sum([df.inner(v, n[k]) * ds(k) for k in marks.get("out")])
        fout = df.assemble(fout)
        v.rename("v", "velocity")
        p.rename("p", "pressure")
        print(
            f"--> report at t= {t: 10.3e} dt= {dt: 10.2e} wtime= {cputime: 10.2e} status= {ts.getConvergedReason()} Flux: in= {fin: 10.2e} out= {fout: 10.2e} wall= {fwall: 10.2e}"
        )
        if xdmf_last and t > t_end:
            ctx["file_xdmf"]["v"].write(v, t)
            ctx["file_xdmf"]["p"].write(p, t)
            # ctx['file_xdmf']['mu'].write(visc_model.viscosity, t)
        elif xdmf_all:
            ctx["file_xdmf"]["v"].write(v, t)
            ctx["file_xdmf"]["p"].write(p, t)
            # ctx['file_xdmf']['mu'].write(visc_model.viscosity, t)
        ctx["file_hdf5"]["w"].write(w, "w", t)
        ctx["file_hdf5"]["w"].write(wdot, "wdot", t)
        ctx["file_hdf5"]["w"].flush()

    if t >= t_end and xdmf_last:
        WW = FE.W
        # ensure P1 recovery even for TH element
        V = df.VectorFunctionSpace(mesh, "P", 1)
        a = df.Function(WW)
        (_, _, v_test, p_test) = FE.split(a)
        g = df.TrialFunction(V)
        g_test = df.TestFunction(V)

        lhs = df.inner(g, g_test) * ds(marks["wall"])

        A = df.assemble(lhs, keep_diagonal=True)
        A.ident_zeros()

        solver_wss = df.KrylovSolver("bicgstab", "jacobi")
        solver_wss.set_operator(A)

        ST = stabilization(
            mesh,
            boundary_parts,
            FE,
            (v, p, vdot, pdot),
            (eqv, eqp),
            (v_test, p_test),
            rho=rho,
            viscosity=visc_model.viscosity,
            k=1.0 / dt,
        )

        F_wss = (
            rho * df.inner(vdot, v_test) * dx
            + rho * df.inner(df.grad(v) * v, v_test) * dx
            + df.inner(T(p, v), df.grad(v_test)) * dx
            + df.div(v) * p_test * dx
            + ST.S
            - (df.inner(T(p, v) * n[marks["in"]], v_test) * ds(marks["in"]))
        )

        # add directional do-nothing BC
        if bcout_dir_do_nothing:
            for j in marks.get("out"):
                F_wss += -0.5 * rho * df.inner(
                    df.conditional(df.gt(vn(v, n[j]), 0.0), 0.0, 1.0) * vn(v, n[j]) * v,
                    v_test,
                ) * ds(j) - (df.inner(T(p, v) * n[j], v_test) * ds(j))

        if Theta == 1.0:

            F_wss += df.inner(
                T(p_test, v_test) * n[marks["wall"]],
                v - df.Constant((0, 0, 0)),
            ) * ds(marks["wall"])

            F_wss += (
                (beta * mu / edgelen)
                * df.inner(v - df.Constant((0, 0, 0)), v_test)
                * ds(marks["wall"])
            )

        # if Theta_in == 1.0:
        #     F_wss += NitscheBC(v - v_in, n[marks["in"]], ds(marks["in"]))

        WFv = block_split(F_wss, 0)
        rhs = df.assemble(df.action(WFv, g_test))
        traction_weak = df.Function(V, name="traction_force")
        solver_wss.solve(traction_weak.vector(), rhs)

        F_wss_tan = F_wss - (
            df.inner(
                df.inner(T(p, v) * n[marks["wall"]], n[marks["wall"]]) * n[marks["wall"]],
                v_test,
            )
            * ds(marks["wall"])
        )
        WFv = block_split(F_wss_tan, 0)
        rhs = df.assemble(df.action(WFv, g_test))
        traction_weak_tan = df.Function(V, name="wss_weak")
        solver_wss.solve(traction_weak_tan.vector(), rhs)

        wss_spaces = [("DG", 0), ("CG", 1)]

        for wss_family, wss_degree in wss_spaces:

            traction_element = df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree)
            traction_space = df.FunctionSpace(mesh, traction_element)
            traction_computer = TractionComputation(traction_space, mark_wall=marks["wall"])

            wss_weak = traction_computer.project_function(traction_weak_tan)

            # standard evaluation
            Tn = T(p, v) * n[marks["wall"]]
            wss = traction_computer.project_wss(Tn, n[marks["wall"]])

            # file_xdmf[f"wss_weak_{wss_family}_{wss_degree}"].write_checkpoint(
            #     wss_weak, "wss", 0
            # )
            # file_xdmf[f"wss_standard_{wss_family}_{wss_degree}"].write_checkpoint(wss, "wss", 0)
            wss_weak.rename("wss", "wss")
            file_xdmf[f"wss_weak_{wss_family}_{wss_degree}"].write(wss_weak)
            wss.rename("wss", "wss")
            file_xdmf[f"wss_standard_{wss_family}_{wss_degree}"].write(wss)


if all_monitor:
    solver_list = [
        "ts",
        "ts_adapt",
        "snes",
        "snes_linesearch",
        "ksp",
        "pc",
        "npc",
        "npc_snes",
        "npc_snes_linesearch",
        "npc_snes_ksp",
        "npc_snes_pc",
    ]
    for s in solver_list:
        opts[s + "_converged_reason"] = ""
        opts[s + "_monitor"] = ""
elif basic_monitor:
    solver_list = ["ts", "ts_adapt", "snes"]
    for s in solver_list:
        opts[s + "_monitor"] = ""


problem = TS.Problem(
    F,
    w,
    wdot,
    bcs,
    J,
    Jwdot,
    update=update,
    report=report,
    prestep=prestep,
    form_compiler_parameters=ffc_opts,
)


# do not recompute Jacobian in every time step
solver_setup = solver_settings.direct_with_lag
# solver_setup=solver_settings.direct # second option when recomputed every time

# use python pretty print to print solver settings

pp = pprint.PrettyPrinter(indent=2)
print(pp.pformat(solver_setup))

ctx = dict()
solver = TS.Solver(problem, setup=solver_setup, context=ctx)
solver.ts.setTime(t_begin)
if uniform_dt_last_period:
    opts["ts_exact_final_time"] = "matchstep"
    solver.ts.setMaxTime(t_period * (periods - 1))
else:
    solver.ts.setMaxTime(t_end)
solver.ts.setTimeStep(dt)

solver.ts.snes.setMonitor(solver_settings.snes_monitor)

solver.ts.setProblemType(solver.ts.ProblemType.NONLINEAR)
solver.ts.setEquationType(solver.ts.EquationType.DAE_IMPLICIT_INDEX2)
solver.ts.setDM(FE.dm)

dm_names, issets, dms = solver.ts.dm.createFieldDecomposition()
# setup error tolerances for the adaptive timestepper
# we want to control the error in v (its evolution eq in v)
# not in p - there is no dp/dt

# set the adaptivity control only on velocity
atol = problem.A_petsc.createVecRight()
rtol = problem.A_petsc.createVecRight()
atol.set(t_atol)  # some kind of absolute tolerance on truncation error
rtol.set(t_rtol)  # some kind of relative tolerance on truncation error
# set the pressure part to 0
atol.setValues(issets[1].array, [0.0] * issets[1].local_size)
rtol.setValues(issets[1].array, [0.0] * issets[1].local_size)
solver.ts.setTolerances(rtol=rtol, atol=atol)
solver.ts.setFromOptions()
solver.ts.setUp()

print(
    f"solving: size= {W.dim()} dof, TSType= {solver.ts.getType()}, element= {ns_element.__name__}"
)

for ci, cname in enumerate(dm_names):
    print(f"component:  {cname}  dofs = {issets[ci].sizes} (local/global)")

script = os.path.basename(__file__)
script_path = os.path.join(folder, script)
if rank == 0:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Restart: use data as initial solution
if restart:
    # move files to a folder called 'before_restart'
    if rank == 0:
        list_dir = os.listdir(folder)
        os.mkdir(os.path.join(folder, "before_restart"))  # make directory

        shutil.move(os.path.join(folder, "w.h5"), os.path.join(folder, "before_restart/w.h5"))
        print("Moved file w.h5")
        list_dir.remove("w.h5")

        if os.path.exists(os.path.join(folder, "v_h5.h5")):  # move file v_h5.h5
            shutil.move(
                os.path.join(folder, "v_h5.h5"), os.path.join(folder, "before_restart/v_h5.h5")
            )
            print("Moved file v_h5.h5")
            list_dir.remove("v_h5.h5")

        # move also all the files in subfolders, for example in computed_quantities_all_*
        for directory in list_dir:
            list_files = os.listdir(os.path.join(folder, directory))
            os.mkdir(os.path.join(folder, "before_restart", directory))

            for file_name in list_files:
                source = os.path.join(folder, directory, file_name)
                destination = os.path.join(folder, "before_restart", directory, file_name)
                shutil.move(source, destination)
                print("Moved:", file_name)
            # remove (empty) directory
            os.rmdir(os.path.join(folder, directory))
    # wait until moving is done
    df.MPI.barrier(comm)

    w0 = df.Function(W)
    w0dot = df.Function(W)
    h5guess = folder + "/before_restart/w.h5"
    h5file = df.HDF5File(comm, h5guess, "r")
    ntimesteps = h5file.attributes("/w")["count"]  # number of time steps
    times = []
    for i in range(ntimesteps):
        times = np.append(
            times, h5file.attributes("/w/vector_%d" % i).to_dict().get("timestamp")
        )
    h5file.read(
        w0, "/w/vector_%d" % (ntimesteps - restart_stepsback)
    )  # to make sure we will start before the end of the last computed period
    h5file.read(w0dot, "/wdot/vector_%d" % (ntimesteps - restart_stepsback))
    df.assign(problem.w, w0)
    df.assign(problem.wdot, w0dot)
    if t_begin == 0:
        t = times[ntimesteps - restart_stepsback]
        if (
            ts_adapt_type != "none"
        ):  # if ts_adapt_type=='none', we do not want to set a different dt
            dt = t - times[ntimesteps - restart_stepsback - 1]
            solver.ts.setTimeStep(dt)
        else:
            t = restart_t_begin
    else:
        # the nearest computed time from t_begin
        t = max([i for i in times if t_begin > i])
        t_idx = times.tolist().index(t)
        if ts_adapt_type != "none":
            dt = times[t_idx] - times[t_idx - 1]  # the nearest dt
            solver.ts.setTimeStep(dt)
    solver.ts.setTime(t)
    h5file.close()
    solver.problem.set_solution()

if xdmf_all:
    resNS_file_xdmf = os.path.join(folder, "results_xdmf", "%s.xdmf")
if xdmf_last:
    resNS_file_xdmf = os.path.join(folder, "last_timestep/%s.xdmf")
resNS_file_hdf5 = os.path.join(folder, "%s.h5")

if xdmf_last or xdmf_all:
    file_xdmf = dict()
file_hdf5 = dict()

if xdmf_last or xdmf_all:
    # for i in ["v", "p"]:  # "mu"
    for i in [
        "v",
        "p",
        "wss_weak_DG_0",
        "wss_standard_DG_0",
        "wss_weak_DG_1",
        "wss_standard_DG_1",
        "wss_weak_CG_1",
        "wss_standard_CG_1",
    ]:  # "mu"
        file_xdmf[i] = df.XDMFFile(comm, resNS_file_xdmf % i)
        file_xdmf[i].parameters["flush_output"] = True
        file_xdmf[i].parameters["rewrite_function_mesh"] = False
file_hdf5["w"] = df.HDF5File(comm, resNS_file_hdf5 % "w", "w")

ctx.update({f: eval(f) for f in ["file_hdf5"]})
if xdmf_last or xdmf_all:
    ctx.update({f: eval(f) for f in ["file_xdmf"]})
ctx.update({"inflow": v_in})
ctx.update({"FE": FE})

opts.view()
report(solver.ts)
solver.solve()

if not uniform_dt_last_period:
    solver.ts.view()
else:
    opts["ts_adapt_type"] = "none"
    solver.ts.setMaxTime(t_end)
    solver.ts.setTimeStep(uniform_dt)
    solver.ts.setFromOptions()
    solver.ts.setUp()
    print(
        "ts adapt type has been changed to none (i.e. uniform length of time steps). New options are the following:"
    )
    opts.view()

    resNS_file_hdf5 = os.path.join(folder, "last_period", "%s.h5")
    if xdmf_last or xdmf_all:
        resNS_file_xdmf = os.path.join(folder, "last_period", "results_xdmf", "%s.xdmf")

    if xdmf_last or xdmf_all:
        file_xdmf = dict()
    file_hdf5 = dict()

    if xdmf_last or xdmf_all:
        for i in ["v", "p"]:  # "mu"
            file_xdmf[i] = df.XDMFFile(comm, resNS_file_xdmf % i)
            file_xdmf[i].parameters["flush_output"] = True
            file_xdmf[i].parameters["rewrite_function_mesh"] = False
    file_hdf5["w"] = df.HDF5File(comm, resNS_file_hdf5 % "w", "w")

    ctx.update({f: eval(f) for f in ["file_hdf5"]})
    if xdmf_last or xdmf_all:
        ctx.update({f: eval(f) for f in ["file_xdmf"]})

    report(solver.ts)
    solver.solve()
    solver.ts.view()

if xdmf_last:
    resNS_file_last = os.path.join(folder, "last_timestep/%s.h5")
    w_last = df.HDF5File(comm, resNS_file_last % "w", "w")
    t = solver.ts.getTime()
    w_last.write(w, "w", t)
    w_last.write(wdot, "wdot", t)
    v, p = w.split(deepcopy=True)
    v.rename("v", "velocity")
    p.rename("p", "pressure")
    # file_xdmf["v"].write(v, t)
    # file_xdmf["p"].write(p, t)
    file_xdmf["v"].write_checkpoint(v, "v", append=False)
    file_xdmf["p"].write_checkpoint(p, "p", append=False)
    # file_xdmf["mu"].write(visc_model.viscosity, t)

if compare_with_poiseuille:
    u_profile = "2.0*v_avg*(pow(r,2)-(pow(x[0],2)+pow(x[1],2)))/(r*r)"
    u_exact = df.Expression(("0.0", "0.0", u_profile), r=0.001, v_avg=v_avg, degree=2)
    V = W.sub(0).collapse()
    u_ex_proj = df.Function(V)
    u_ex_proj = df.project(u_exact, V)

    file_xdmf["v_exact"] = df.XDMFFile(comm, os.path.join(folder, "last_timestep/v_exact.xdmf"))
    file_xdmf["v_exact"].parameters["rewrite_function_mesh"] = False
    u_ex_proj.rename("v", "velocity")
    t = solver.ts.getTime()
    file_xdmf["v_exact"].write(u_ex_proj, t)

    v_max = 2 * v_avg
    delta_p = 4 * mu * v_max * 0.002 / 0.001**2
    p_exact = df.Expression("P*(1-x[2]/L)", P=delta_p, L=0.002, degree=2)

    error_velocity = np.sqrt(df.assemble(df.dot(v - u_exact, v - u_exact) * dx))
    error_pressure = np.sqrt(df.assemble(df.dot(p - p_exact, p - p_exact) * dx))
    print(f"L2 error velocity = {error_velocity}")
    print(f"L2 error pressure = {error_pressure}")
    with open(os.path.join(folder, "last_timestep/L2_error.txt"), "w") as f:
        f.write("L2 norm of the velocity difference wrt analytical solution\n")
        f.write(str(error_velocity))
        f.write("\n")
        f.write(str(error_pressure))


file_hdf5["w"].close()
print(df.list_timings(df.TimingClear.keep, [df.TimingType.wall]))
