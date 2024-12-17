import argparse

import dolfin as df
import numpy as np


def parse_edgelengths(edgelengths_str):
    # Split the string by commas and convert each item to an integer
    return [int(x) for x in edgelengths_str.split(",")]


desc = "Compute minimum, maximum, average WSS and LSA for challenge case01 or 02, th or p1p1 element, and a list of edgelengths."
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
    "--mesh-folder",
    default="",
    type=str,
    dest="mesh_folder",
    help="path to the corresponding mesh marked by facet markers (.h5)",
)
parser.add_argument(
    "--res-folder",
    default="",
    type=str,
    dest="res_folder",
    help="path to the results",
)
parser.add_argument(
    "--element",
    default="p1p1",
    type=str,
    choices=["p1p1", "th"],
    dest="element",
    help="finite element; p1p1 or th",
)
parser.add_argument(
    "--stab",
    default="ip",
    type=str,
    choices=["ip", "none"],
    dest="stab",
    help="stabilization; ip for p1p1, none for th",
)
parser.add_argument(
    "--edgelengths",
    default="300,250,200,150,100",
    type=parse_edgelengths,  # Use the custom parser function
    dest="edgelengths",
    help="A comma-separated list of edgelengths in micrometers, e.g., 300,250,200,150,100",
)

args = parser.parse_args()
case = args.case
element = args.element
stab = args.stab
edgelengths = args.edgelengths
print(f"{case=}")
print(f"{element=}")
print(f"{stab=}")
print(f"{edgelengths=}")

marker_dome = 42
marker_parent_artery = 73


for edgelength in edgelengths:

    print(f"edge length = {edgelength*1e-3} mm")

    # read mesh
    mesh_folder = args.mesh_folder
    res_folder = args.res_folder

    mesh = df.Mesh()
    with df.HDF5File(df.MPI.comm_world, mesh_folder, "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        dim = mesh.geometry().dim()
        boundary_parts = df.MeshFunction("size_t", mesh, dim - 1, 0)
        hdf.read(boundary_parts, "/boundaries")
    mesh.init()

    if case == "case01":
        # mark aneurysm dome and parent artery
        origin = [0.0002, -0.00105, 0.00113]
        normal = [0.105, -0.909, -0.403]

        origin_parent_in = [-0.0043, 0.0113, 0.0076]
        origin_parent_out = [-0.00024, 0.0032, 0.0037]
        normal_parent = [0.397, -0.826, -0.401]

        # Mark the subdomain using a MeshFunction
        facet_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        # Mark the aneurysm dome with a predefined marker
        for f in df.facets(mesh, "all"):  # facet=triangle in 3D
            if f.exterior():  # if the facet is exterior
                facet_marker[f] = 0  # label with 0
                s = f.midpoint()  # center of the triangle
                # Compute signed distance from the midpoint to the plane
                distance = (
                    (s[0] - origin[0]) * normal[0]
                    + (s[1] - origin[1]) * normal[1]
                    + (s[2] - origin[2]) * normal[2]
                )
                if distance >= 0:
                    facet_marker[f] = marker_dome
                distance_parent_out = (
                    (s[0] - origin_parent_out[0]) * normal_parent[0]
                    + (s[1] - origin_parent_out[1]) * normal_parent[1]
                    + (s[2] - origin_parent_out[2]) * normal_parent[2]
                )
                distance_parent_in = (
                    (s[0] - origin_parent_in[0]) * normal_parent[0]
                    + (s[1] - origin_parent_in[1]) * normal_parent[1]
                    + (s[2] - origin_parent_in[2]) * normal_parent[2]
                )
                if distance_parent_out <= 0 and distance_parent_in >= 0:
                    facet_marker[f] = marker_parent_artery
    elif case == "case02":
        # mark aneurysm dome and parent artery
        origin1 = [0.00321, -0.00139, -0.00589]
        normal1 = [0.135, -0.429, -0.893]

        origin2 = [0.005, -0.003, -0.002]
        normal2 = [0.5, -0.6, 0.6]

        origin_parent_in = [-0.0089, 0.00122, -0.00108]
        origin_parent_out = [-0.0021, 0.0041, -0.0065]
        normal_parent_in = [0.74, 0.323, -0.589]
        normal_parent_out = [0.715, 0.101, -0.69]

        # Mark the subdomain using a MeshFunction
        facet_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        # Mark the aneurysm dome with a predefined marker
        for f in df.facets(mesh, "all"):  # facet=triangle in 3D
            if f.exterior():  # if the facet is exterior
                facet_marker[f] = 0  # label with 0
                s = f.midpoint()  # center of the triangle
                # Compute signed distance from the midpoint to plane 1
                distance1 = (
                    (s[0] - origin1[0]) * normal1[0]
                    + (s[1] - origin1[1]) * normal1[1]
                    + (s[2] - origin1[2]) * normal1[2]
                )
                # Compute signed distance from the midpoint to plane 2
                distance2 = (
                    (s[0] - origin2[0]) * normal2[0]
                    + (s[1] - origin2[1]) * normal2[1]
                    + (s[2] - origin2[2]) * normal2[2]
                )
                if (distance1 >= 0) and (distance2 <= 0):
                    facet_marker[f] = marker_dome

                distance_parent_out = (
                    (s[0] - origin_parent_out[0]) * normal_parent_out[0]
                    + (s[1] - origin_parent_out[1]) * normal_parent_out[1]
                    + (s[2] - origin_parent_out[2]) * normal_parent_out[2]
                )
                distance_parent_in = (
                    (s[0] - origin_parent_in[0]) * normal_parent_in[0]
                    + (s[1] - origin_parent_in[1]) * normal_parent_in[1]
                    + (s[2] - origin_parent_in[2]) * normal_parent_in[2]
                )
                if distance_parent_out <= 0 and distance_parent_in >= 0:
                    facet_marker[f] = marker_parent_artery
    else:
        raise ValueError("Cannot mark the geometry. Choose either case01 or case02")

    dx = df.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
    ds = df.Measure(
        "ds", domain=mesh, subdomain_data=facet_marker, metadata={"quadrature_degree": 4}
    )
    dS = df.Measure("dS", domain=mesh, metadata={"quadrature_degree": 4})

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

    def find_extremes(filename, wss_family, wss_degree):

        wss_fun = df.Function(
            df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
        )

        file_xdmf = df.XDMFFile(df.MPI.comm_world, filename)
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False
        file_xdmf.read_checkpoint(wss_fun, "wss", 0)

        # Save the subdomain marker to an XDMF file
        with df.XDMFFile(mesh.mpi_comm(), f"{res_folder}/facet_marker.xdmf") as xdmf_file:
            xdmf_file.write(facet_marker)

        # Initialize min and max values
        min_value = float("inf")
        max_value = float("-inf")

        # Loop over each facet in the mesh and check if it is marked with the predefined marker
        for facet in df.facets(mesh):
            if (
                facet_marker[facet.index()] == marker_dome
            ):  # Check if the facet is in the desired subdomain

                if wss_degree == 0:

                    # Evaluate the DG0 function on this cell (single value per cell)
                    facet_value = wss_fun(
                        facet.midpoint()
                    )  # Evaluate the DG0 function, which is constant in the cell
                    facet_magnitude = np.linalg.norm(
                        facet_value
                    )  # Compute the magnitude of the vector (if vector-valued)

                    # Update min and max values
                    if facet_magnitude < min_value:
                        min_value = facet_magnitude
                    if facet_magnitude > max_value:
                        max_value = facet_magnitude

                else:
                    # Evaluate the function at each vertex of the facet
                    for vertex in df.vertices(facet):
                        point_value = wss_fun(vertex.point().array())
                        point_magnitude = np.linalg.norm(
                            point_value
                        )  # Compute the magnitude of the vector

                        # Update min and max values
                        if point_magnitude < min_value and point_magnitude > 0.0:
                            min_value = point_magnitude
                        if point_magnitude > max_value:
                            max_value = point_magnitude

        return min_value, max_value

    def integrate_wss_on_subdomain(filename, wss_family, wss_degree, marker_subdomain):
        # Define the function space and read the wss function data
        wss_fun = df.Function(
            df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
        )

        file_xdmf = df.XDMFFile(df.MPI.comm_world, filename)
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False
        file_xdmf.read_checkpoint(wss_fun, "wss", 0)

        # # Initialize integration variables
        wss_mag = compute_norm(wss_fun)

        total_magnitude = df.assemble(wss_mag * ds(marker_subdomain))
        total_area = df.assemble(df.Constant(1.0) * ds(marker_subdomain))
        average_wss = total_magnitude / total_area

        return average_wss

    def compute_LSA(filename, wss_family, wss_degree, wss_avg, marker_dome):

        low_shear = 0.1 * wss_avg

        # Define the function space and read the wss function data
        wss_fun = df.Function(
            df.FunctionSpace(mesh, df.VectorElement(wss_family, mesh.ufl_cell(), wss_degree))
        )

        file_xdmf = df.XDMFFile(df.MPI.comm_world, filename)
        file_xdmf.parameters["flush_output"] = True
        file_xdmf.parameters["rewrite_function_mesh"] = False
        file_xdmf.read_checkpoint(wss_fun, "wss", 0)

        # # Initialize integration variables
        wss_mag = compute_norm(wss_fun)

        # Define and compute the low shear area
        LSA_fun = df.Function(
            df.FunctionSpace(mesh, df.FiniteElement(wss_family, mesh.ufl_cell(), wss_degree))
        )

        LSA_fun.vector()[:] = np.where(wss_mag.vector()[:] > low_shear, 0, 1)
        int_LSA = df.assemble(LSA_fun * ds(marker_dome))
        total_area = df.assemble(df.Constant(1.0) * ds(marker_dome))
        LSA = int_LSA / total_area * 100

        return LSA

    print("========================")
    print("BOUNDARY-FLUX EVALUATION")
    print("========================")
    wss_family = "CG"
    wss_degree = 1

    filename = f"{res_folder}/wss_weak.xdmf"

    minimum_weak_CG1, maximum_weak_CG1 = find_extremes(filename, wss_family, wss_degree)
    average_wss_weak_CG1 = integrate_wss_on_subdomain(
        filename, wss_family, wss_degree, marker_dome
    )
    average_parent_wss_weak_CG1 = integrate_wss_on_subdomain(
        filename, wss_family, wss_degree, marker_parent_artery
    )
    LSA_weak = compute_LSA(
        filename, wss_family, wss_degree, average_parent_wss_weak_CG1, marker_dome
    )
    print(f"maximum WSS boundary-flux: {maximum_weak_CG1:.2f} Pa")
    print(f"minimum WSS boundary-flux: {minimum_weak_CG1:.3f} Pa")
    print(f"average WSS boundary-flux: {average_wss_weak_CG1:.2f} Pa")
    print(f"LSA boundary-flux: {LSA_weak:.2f}%")

    print("========================")
    print("P1, DG1 AND DG0 PROJECTION")
    print("========================")

    filename = f"{res_folder}/wss_standard_{wss_family}_{wss_degree}.xdmf"
    minimum_CG1, maximum_CG1 = find_extremes(filename, wss_family, wss_degree)
    average_wss_CG1 = integrate_wss_on_subdomain(filename, wss_family, wss_degree, marker_dome)
    average_parent_wss_CG1 = integrate_wss_on_subdomain(
        filename, wss_family, wss_degree, marker_parent_artery
    )
    LSA_CG1 = compute_LSA(filename, wss_family, wss_degree, average_parent_wss_CG1, marker_dome)
    print(f"maximum WSS P1 projection: {maximum_CG1:.2f} Pa")
    print(f"minimum WSS P1 projection: {minimum_CG1:.3f} Pa")
    print(f"average WSS P1 projection: {average_wss_CG1:.2f} Pa")
    print(f"LSA P1 projection: {LSA_CG1:.2f}%")

    print("========================")
    wss_family = "DG"
    wss_degree = 1
    filename = f"{res_folder}/wss_standard_{wss_family}_{wss_degree}.xdmf"
    minimum_DG1, maximum_DG1 = find_extremes(filename, wss_family, wss_degree)
    average_wss_DG1 = integrate_wss_on_subdomain(filename, wss_family, wss_degree, marker_dome)
    average_parent_wss_DG1 = integrate_wss_on_subdomain(
        filename, wss_family, wss_degree, marker_parent_artery
    )
    LSA_DG1 = compute_LSA(filename, wss_family, wss_degree, average_parent_wss_DG1, marker_dome)
    print(f"maximum WSS DG1 projection: {maximum_DG1:.2f} Pa")
    print(f"minimum WSS DG1 projection: {minimum_DG1:.3f} Pa")
    print(f"average WSS DG1 projection: {average_wss_DG1:.2f} Pa")
    print(f"LSA DG1 projection: {LSA_DG1:.2f}%")

    print("========================")
    wss_family = "DG"
    wss_degree = 0
    filename = f"{res_folder}/wss_standard_{wss_family}_{wss_degree}.xdmf"
    minimum_DG0, maximum_DG0 = find_extremes(filename, wss_family, wss_degree)
    average_wss_DG0 = integrate_wss_on_subdomain(filename, wss_family, wss_degree, marker_dome)
    average_parent_wss_DG0 = integrate_wss_on_subdomain(
        filename, wss_family, wss_degree, marker_parent_artery
    )
    LSA_DG0 = compute_LSA(filename, wss_family, wss_degree, average_parent_wss_DG0, marker_dome)
    print(f"maximum WSS DG0 projection: {maximum_DG0:.2f} Pa")
    print(f"minimum WSS DG0 projection: {minimum_DG0:.3f} Pa")
    print(f"average WSS DG0 projection: {average_wss_DG0:.2f} Pa")
    print(f"LSA DG0 projection: {LSA_DG0:.2f}%")
