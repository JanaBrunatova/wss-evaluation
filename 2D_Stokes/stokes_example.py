import argparse

import dolfin
import numpy as np
import ufl

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

dolfin.parameters["std_out_all_processes"] = False
dolfin.parameters["allow_extrapolation"] = False
dolfin.parameters["ghost_mode"] = "shared_vertex"

parser = argparse.ArgumentParser(
    description="Solve P1-P1 Stokes with edge stabilization; problem taken from Burman and Hansbo doi:10.1016/j.cma.2005.05.009",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--N",
    "-N",
    dest="N",
    nargs="+",
    type=int,
    help="Number of elements in each direction",
    required=True,
)
parser.add_argument(
    "--gamma",
    "-g",
    dest="gamma",
    default=0.01,
    type=float,
    help="Stabilization parameter. Will be used only for p1p1 element, not for th.",
)
parser.add_argument(
    "--lambda",
    dest="lmbda",
    default=1000.0,
    type=float,
    help="Stabilization parameter. Will be used only for p1p1 element, not for th.",
)
parser.add_argument("--C", "-C", dest="C", default=-5.0, type=float, help="Pressure constant")
parser.add_argument("--nu", dest="nu", default=1.0, type=float, help="Viscosity")

parser.add_argument(
    "-o",
    default="results_stokes_p1p1/",
    type=str,
    dest="res_folder",
    help="results folder",
)
parser.add_argument(
    "--mesh-type",
    default="crossed",
    type=str,
    dest="mesh_type",
    choices=["right", "crossed", "left"],
    help="RectangleMesh type",
)
parser.add_argument(
    "--element",
    default="p1p1",
    type=str,
    choices=["p1p1", "th"],
    dest="elem",
    help="finite element; p1p1 or th",
)


def compute_rate(h, error):
    return np.log(error[1:] / error[:-1]) / np.log(h[1:] / h[:-1])


def T(v, p, nu):  # Cauchy stress tensor
    return -p * dolfin.Identity(v.ufl_shape[0]) + nu * ufl.grad(v)


def ET(v, p, nu):  # Cauchy stress tensor
    return nu * ufl.grad(v)


def tangential_proj(v, n):
    """
    Compute the tangential projection of a vector v given the normal vector n.
    """
    return (dolfin.Identity(v.ufl_shape[0]) - dolfin.outer(n, n)) * v


def wss_expr(v, p, nu, n):
    Tn = ufl.dot(T(v, p, nu), n)
    # return Tn - ufl.dot(Tn, n) * n
    return tangential_proj(Tn, n)


def v_ex(x):
    return 20 * x[0] * x[1] ** 3, 5 * x[0] ** 4 - 5 * x[1] ** 4


def p_ex(C, x):
    return 60 * x[0] ** 2 * x[1] - 20 * x[1] ** 3 + C


if __name__ == "__main__":
    args = parser.parse_args()
    nu = args.nu
    elem = args.elem

    h_arr = []
    errors_v = []
    errors_p = []
    errors_wss_dg0 = []
    errors_wss_dg1 = []
    errors_wss_cg1 = []
    errors_wss_weak = []

    for i, N in enumerate(args.N):
        # Create mesh
        mesh = dolfin.UnitSquareMesh(N, N, args.mesh_type)
        mesh.init()

        # Mark boundaries split according to jump in normal
        wall_marks = [1, 2, 3, 4]
        boundary = dolfin.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, 0)
        for f in dolfin.facets(mesh):
            if f.exterior():
                x = f.midpoint()
                if dolfin.near(x[1], 1.0):
                    boundary[f] = 1
                if dolfin.near(x[0], 1.0):
                    boundary[f] = 2
                if dolfin.near(x[1], 0.0):
                    boundary[f] = 3
                if dolfin.near(x[0], 0.0):
                    boundary[f] = 4

        # dolfin.File("boundary.pvd") << boundary

        # Use Facet normal
        n = ufl.FacetNormal(mesh)

        # Define integration measures
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 7})

        ds = ufl.Measure(
            "ds", domain=mesh, subdomain_data=boundary, metadata={"quadrature_degree": 7}
        )

        dS = ufl.Measure("dS", domain=mesh, metadata={"quadrature_degree": 7})

        # Define the right-hand side
        x = dolfin.SpatialCoordinate(mesh)
        f = -ufl.div(T(ufl.as_vector(v_ex(x)), p_ex(args.C, x), nu))

        # Define unknowns and test function(s) P1P1 element
        if elem == "p1p1":
            V = dolfin.VectorElement("P", mesh.ufl_cell(), 1)
            P = dolfin.FiniteElement("P", mesh.ufl_cell(), 1)
        else:
            V = dolfin.VectorElement("P", mesh.ufl_cell(), 2)
            P = dolfin.FiniteElement("P", mesh.ufl_cell(), 1)
        W = dolfin.FunctionSpace(mesh, dolfin.MixedElement([V, P]))
        w = dolfin.Function(W)
        v, p = dolfin.split(w)
        v_, p_ = dolfin.TestFunctions(W)

        # h = dolfin.MinCellEdgeLength(mesh)
        h = dolfin.Constant(2.0) * ufl.Circumradius(mesh)

        s = dolfin.conditional(
            dolfin.gt(nu, dolfin.avg(h)), dolfin.Constant(2.0), dolfin.Constant(1.0)
        )

        def j(p, q):
            return (
                0.5
                * args.gamma
                * dolfin.avg(h) ** (1 + s)
                * dolfin.inner(dolfin.jump(dolfin.grad(p), n), dolfin.jump(dolfin.grad(q), n))
                * dS
            )

        def j_tilde(u, v):
            return (
                0.5
                * args.gamma
                * dolfin.avg(h) ** (1 + s)
                * dolfin.inner(dolfin.jump(dolfin.div(u)), dolfin.jump(dolfin.div(v)))
                * dS
            )

        # Weak formulation
        FF = (
            ufl.inner(T(v, p, nu), ufl.grad(v_)) * dx
            + ufl.div(v) * p_ * dx
            - ufl.inner(f, v_) * dx
        )

        if elem == "p1p1":
            FF += j_tilde(v, v_) + j(p, p_)

        def NitscheBC(
            eq, flux, w, ds, beta=1e3
        ):  # Nitsche's nonsymmetric method implemented through the derivative of a functional
            w_ = dolfin.TestFunction(w.function_space())
            penalty = (beta / h) * dolfin.inner(eq, dolfin.derivative(eq, w, w_)) * ds
            bcpart = (
                dolfin.inner(flux, dolfin.derivative(eq, w, w_)) * ds
                - dolfin.inner(dolfin.derivative(flux, w, w_), eq) * ds
            )
            return -bcpart + penalty

        u_bc = ufl.as_vector(v_ex(x))

        class InitExpression(dolfin.UserExpression):
            def __init__(self, u_bc, **kwargs):
                super().__init__(**kwargs)
                self.u = u_bc

            def eval(self, values, x):
                values[0] = self.u[0](x)
                values[1] = self.u[1](x)

            def value_shape(self):
                return (2,)

        u_bc = InitExpression(u_bc)

        if elem == "p1p1":
            # Nitsche BC
            lmbda = dolfin.Constant(args.lmbda)
            Fbc = [
                NitscheBC(v - u_bc, T(v, p, nu) * n, w, ds(ii), beta=lmbda) for ii in wall_marks
            ]
            F = FF + sum(Fbc)
            bcs = []
        else:
            bcs = []
            for ii in wall_marks:
                bc_wall = dolfin.DirichletBC(W.sub(0), u_bc, boundary, ii)
                bcs.append(bc_wall)
            F = FF

        problem = dolfin.NonlinearVariationalProblem(F, w, bcs, J=dolfin.derivative(F, w))
        solver = dolfin.NonlinearVariationalSolver(problem)

        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-8
        solver.solve()

        vv, pp = w.split(deepcopy=True)

        # Normalize pressure
        p_avg = dolfin.assemble(pp * dx)
        p_ex_avg = dolfin.assemble(p_ex(args.C, x) * dx)
        pp.vector()[:] -= p_avg + p_ex_avg

        # Save solution
        vv.rename("v", "velocity")
        pp.rename("p", "pressure")
        with dolfin.XDMFFile(f"{args.res_folder}v_N{N}.xdmf") as xdmf:
            xdmf.write_checkpoint(vv, "v", 0.0, append=False)
        with dolfin.XDMFFile(f"{args.res_folder}p_N{N}.xdmf") as xdmf:
            xdmf.write_checkpoint(pp, "p", 0.0, append=False)

        # Compute errors
        v_error = np.sqrt(
            dolfin.assemble(
                dolfin.dot((vv - ufl.as_vector(v_ex(x))), (vv - ufl.as_vector(v_ex(x)))) * dx
            )
        )
        p_error = np.sqrt(
            dolfin.assemble(dolfin.dot((pp - p_ex(args.C, x)), (pp - p_ex(args.C, x))) * dx)
        )

        # WSS computations
        # ================

        def standard_evaluation(traction_space, wall_marks, v, p, nu, n, filename):
            t = dolfin.TrialFunction(traction_space)
            t_ = dolfin.TestFunction(traction_space)
            solver_wss = dolfin.LUSolver("mumps")

            wss_set = {}
            for ii in wall_marks:
                # Define the left-hand side
                lhs = ufl.inner(t, t_) * ds(ii)
                A = dolfin.assemble(lhs, keep_diagonal=True)
                A.ident_zeros()
                solver_wss.set_operator(A)

                rhs = ufl.inner(wss_expr(v, p, nu, n), t_) * ds(ii)
                b = dolfin.assemble(rhs)
                wss = dolfin.Function(traction_space)
                solver_wss.solve(wss.vector(), b)
                wss.rename("wss", "wss")
                wss_set[ii] = wss

                # save the solution on the upper boundary
                if ii == 1:
                    with dolfin.XDMFFile(filename) as xdmf:
                        xdmf.write_checkpoint(wss, "wss", 0.0, append=False)
            return wss_set

        # Standard evaluation CG1
        traction_element = dolfin.VectorElement("CG", mesh.ufl_cell(), 1)
        traction_space = dolfin.FunctionSpace(mesh, traction_element)
        wss_set_cg1 = standard_evaluation(
            traction_space,
            wall_marks,
            v,
            p,
            nu,
            n,
            f"{args.res_folder}wss_N{N}_standard_CG1.xdmf",
        )

        # Standard evaluation DG1
        traction_element = dolfin.VectorElement("DG", mesh.ufl_cell(), 1)
        traction_space = dolfin.FunctionSpace(mesh, traction_element)
        wss_set_dg1 = standard_evaluation(
            traction_space,
            wall_marks,
            v,
            p,
            nu,
            n,
            f"{args.res_folder}wss_N{N}_standard_DG1.xdmf",
        )

        # Standard evaluation DG0
        traction_element = dolfin.VectorElement("DG", mesh.ufl_cell(), 0)
        traction_space = dolfin.FunctionSpace(mesh, traction_element)
        wss_set_dg0 = standard_evaluation(
            traction_space,
            wall_marks,
            v,
            p,
            nu,
            n,
            f"{args.res_folder}wss_N{N}_standard_DG0.xdmf",
        )

        # Weak evaluation
        traction_element_weak = dolfin.VectorElement("CG", mesh.ufl_cell(), 1)
        traction_space_weak = dolfin.FunctionSpace(mesh, traction_element_weak)
        # traction_space_weak = w.sub(0).function_space().collapse()
        g = dolfin.TrialFunction(traction_space_weak)
        g_test = dolfin.TestFunction(traction_space_weak)
        solver_wss_weak = dolfin.LUSolver("mumps")

        wss_weak_set = {}
        if elem == "p1p1":
            for ii in wall_marks:
                Fbc = sum(
                    [
                        NitscheBC(v - u_bc, T(v, p, nu) * n, w, ds(j), beta=0.0)
                        for j in wall_marks
                        if j != ii
                    ]
                )

                F_wss = FF + Fbc - ufl.inner(ufl.inner(T(v, p, nu) * n, n) * n, v_) * ds(ii)

                WFv = ufl.block_split(F_wss, 0)
                # Define the left-hand side
                lhs = ufl.inner(g, g_test) * ds(ii)
                rhs = ufl.action(WFv, g_test)
                A = dolfin.assemble(lhs, keep_diagonal=True)
                A.ident_zeros()
                solver_wss_weak.set_operator(A)

                b = dolfin.assemble(rhs)
                wss_weak = dolfin.Function(traction_space_weak)
                wss_weak.rename("wss", "wss")
                solver_wss_weak.solve(wss_weak.vector(), b)
                wss_weak_set[ii] = wss_weak

                # save the solution on the upper boundary
                if ii == 1:
                    with dolfin.XDMFFile(f"{args.res_folder}wss_N{N}_weak_CG1.xdmf") as xdmf:
                        xdmf.write_checkpoint(wss_weak, "wss", 0.0, append=False)
        else:
            for ii in wall_marks:

                F_wss = (
                    FF
                    - ufl.inner(ufl.inner(T(v, p, nu) * n, n) * n, v_) * ds(ii)
                    - sum(
                        [
                            (ufl.inner(T(v, p, nu) * n, v_) * ds(j))
                            for j in wall_marks
                            if j != ii
                        ]
                    )  # subtract Neumann BC applied on the other three sides of the square
                )

                WFv = ufl.block_split(F_wss, 0)
                # Define the left-hand side
                lhs = ufl.inner(g, g_test) * ds(ii)
                rhs = ufl.action(WFv, g_test)
                A = dolfin.assemble(lhs, keep_diagonal=True)
                b = dolfin.assemble(rhs)

                A.ident_zeros()

                solver_wss_weak.set_operator(A)

                wss_weak = dolfin.Function(traction_space_weak)
                wss_weak.rename("wss", "wss")
                solver_wss_weak.solve(wss_weak.vector(), b)
                wss_weak_set[ii] = wss_weak

                # save the solution on the upper boundary
                if ii == 1:
                    with dolfin.XDMFFile(f"{args.res_folder}wss_N{N}_weak_CG1.xdmf") as xdmf:
                        xdmf.write_checkpoint(wss_weak, "wss", 0.0, append=False)

        # Exact traction
        wss_exact = wss_expr(ufl.as_vector(v_ex(x)), p_ex(args.C, x), nu, n)

        def l2_norm2(f, ds):
            return dolfin.dot(f, f) * ds

        h_error_weak = np.sqrt(
            sum(
                [
                    dolfin.assemble(l2_norm2(wss_weak_set[ii] - wss_exact, ds(ii)))
                    for ii in wall_marks
                ]
            )
        )
        h_error_dg1 = np.sqrt(
            sum(
                [
                    dolfin.assemble(l2_norm2(wss_set_dg1[ii] - wss_exact, ds(ii)))
                    for ii in wall_marks
                ]
            )
        )
        h_error_dg0 = np.sqrt(
            sum(
                [
                    dolfin.assemble(l2_norm2(wss_set_dg0[ii] - wss_exact, ds(ii)))
                    for ii in wall_marks
                ]
            )
        )
        h_error_cg1 = np.sqrt(
            sum(
                [
                    dolfin.assemble(l2_norm2(wss_set_cg1[ii] - wss_exact, ds(ii)))
                    for ii in wall_marks
                ]
            )
        )

        if rank == 0:
            print(f"{N=}")
            print(f"velocity error: {v_error}")
            print(f"pressure error: {p_error}")
            print(f"wss standard DG1 error: {h_error_dg1}")
            print(f"wss standard DG0 error: {h_error_dg0}")
            print(f"wss standard CG1 error: {h_error_cg1}")
            print(f"wss weak error: {h_error_weak}")

        h_arr.append(1 / N)
        errors_v.append(v_error)
        errors_p.append(p_error)
        errors_wss_dg1.append(h_error_dg1)
        errors_wss_dg0.append(h_error_dg0)
        errors_wss_cg1.append(h_error_cg1)
        errors_wss_weak.append(h_error_weak)

    if len(errors_v) > 1:

        rate_v = compute_rate(np.array(h_arr), np.array(errors_v))
        rate_p = compute_rate(np.array(h_arr), np.array(errors_p))
        rate_wss_dg1 = compute_rate(np.array(h_arr), np.array(errors_wss_dg1))
        rate_wss_dg0 = compute_rate(np.array(h_arr), np.array(errors_wss_dg0))
        rate_wss_cg1 = compute_rate(np.array(h_arr), np.array(errors_wss_cg1))
        rate_wss_weak = compute_rate(np.array(h_arr), np.array(errors_wss_weak))

        if rank == 0:
            print("========================")
            print(f"rate velocity: {rate_v.tolist()}")
            print(f"rate pressure: {rate_p.tolist()}")
            print(f"rate wss DG1 standard: {rate_wss_dg1.tolist()}")
            print(f"rate wss DG0 standard: {rate_wss_dg0.tolist()}")
            print(f"rate wss CG1 standard: {rate_wss_cg1.tolist()}")
            print(f"rate wss weak: {rate_wss_weak.tolist()}")
            print("========================")
            print(f"v errors: {errors_v}")
            print(f"p errors: {errors_p}")
            print(f"wss DG1 errors: {errors_wss_dg1}")
            print(f"wss DG0 errors: {errors_wss_dg0}")
            print(f"wss CG1 errors: {errors_wss_cg1}")
            print(f"wss weak errors: {errors_wss_weak}")
