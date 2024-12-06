#  Numerical evaluation of wall shear stress using FEM

# Prerequisities

The code is written in Python3 and requires the following:

1. **Legacy FEniCS 2019.1.0**  
   Installation guide can be found at [FEniCS Project Archive](https://fenicsproject.org/download/archive/).

2. **`ns_aneurysm` Package** (for 3D simulations) 
   The package can be found on our [GitLab repository](https://gitlab.karlin.mff.cuni.cz/bio/aneurysm). To install it, use the following command:
   ```bash
   python3 -m pip install -e .
    ```

# 2D Stokes example

Results for the 2D Stokes flow on a unit square can be reproduced by running the following commands:
- For the P1/P1 stabilized element:
```bash
python3 python3 stokes_example.py -N 8 16 32 64 128 256 512 -g 0.01 --lambda 10 -o res_stokes_p1p1/ --element p1p1
```
- For the P2/P1 element:
```bash
python3 python3 stokes_example.py -N 8 16 32 64 128 256 512 -o res_stokes_th/ --element th
```


# 3D Poiseuille flow

To run the 3D Poiseuille flow example, use the following command:

```bash
python3 poiseuille_flow.py -model Newtonian -rho 1000 -mu 0.004 -theta 1.0 -unit_system SI -meshname ${meshname} -meshfolder ${meshfolder} -element ${element} -normal FacetNormal -basic_monitor -refsys_filename meshes/cylinder_refsys.dat -v_avg 0.5 -xdmf_last True -dest ${destination}
```

**Parameters to set:**

- `meshname`: Name of the mesh file.
- `meshfolder`: Folder containing the mesh file.
- `element`: Element type (either `p1p1` or `th`).
- `destination`: Name of the folder where the results will be stored.

# 3D Patient-specific simulations

TODO
