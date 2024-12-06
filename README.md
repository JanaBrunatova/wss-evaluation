#  Numerical evaluation of wall shear stress using FEM

This repository contain codes and necessary files to reproduce data published in the paper
**_On the numerical evaluation of wall shear stress using the finite element method_**

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

Results for the 2D Stokes flow on a unit square can be reproduced by running the following commands in the folder `2D_Stokes`:
- For the P1/P1 stabilized element:
```bash
python3 python3 stokes_example.py -N 8 16 32 64 128 256 512 -g 0.01 --lambda 10 -o res_stokes_p1p1/ --element p1p1
```
- For the P2/P1 element:
```bash
python3 python3 stokes_example.py -N 8 16 32 64 128 256 512 -o res_stokes_th/ --element th
```


# 3D Poiseuille flow

To run the 3D Poiseuille flow example, go to the folder `3D_Poiseuille` and use the following command:

```bash
python3 poiseuille_flow.py -model Newtonian -rho 1000 -mu 0.004 -theta 1.0 -unit_system SI -meshname ${meshname} -meshfolder ${meshfolder} -element ${element} -normal FacetNormal -basic_monitor -refsys_filename meshes/cylinder_refsys.dat -v_avg 0.5 -xdmf_last True -dest ${destination}
```

**Parameters to set:**

- `meshname`: Name of the mesh file.
- `meshfolder`: Folder containing the mesh file.
- `element`: Element type (either `p1p1` or `th`).
- `destination`: Name of the folder where the results will be stored.

# 3D Patient-specific simulations

To run the 3D patient-specific simulations, use the following command with similar parameters as described above and one additional:
- `case`: either `case01` or `case02`

```bash
python3 aneurysm_example.py -model Newtonian -mu 0.004 -rho 1000 -theta -1.0 -unit_system SI -meshname ${meshname} -meshfolder ${meshfolder} -element th -normal FacetNormal -stab none -basic_monitor -refsys_filename meshes/${case}_refsystems_SI.dat -profile stac -profile_analytical True -v-avg 0.5 -bcout_dir_do_nothing False -unit_system SI -xdmf_last True -dest ${destination}
```

- For the P1/P1 stabilized element:
```bash
python aneurysm_example.py -model Newtonian -mu 0.004 -rho 1000 -theta 1.0 -theta_in 1.0 -beta 100 -unit_system SI -meshname ${meshname} -meshfolder ${meshfolder} -element p1p1 -normal FacetNormal -stab ip -basic_monitor -refsys_filename meshes/${case}_refsystems_SI.dat -profile stac -profile_analytical True -v-avg 0.5 -bcout_dir_do_nothing False -unit_system SI -xdmf_last True -dest ${destination}
```

It will store a HDF5File `w.h5`, where both the velocity and pressure fields are stored. Moreover, since we used the option `-xdmf_last True`, the velocity and pressure will also be stored in an XDMF file format.

Subsequently, go to the folder `3D_patient_specific` and run the following postprocessing to obtain WSS from the velocity and pressure field:

- For the P2/P1 element:
```bash
python3 compute_wss_aneurysm.py --element ${element} --mesh ${meshpath} --case ${case} -o ${output_folder} --w-file ${path_to_w_file} --compute-differences --stationary
```

- For the P1/P1 stabilized element:
```bash
python compute_wss_aneurysm.py --element ${element} --stab ip --mesh ${meshpath} --case ${case} -o ${output_folder} --w-file ${path_to_w_file} --compute-differences --stationary --nitsche-noslip --beta 100
```

**Parameters to set:**

- `meshpath`: Name of the mesh file containing the folder as well.
- `output_folder`: Name of the output folder where wall shear stresses will be stored.
- `path_to_w_file`: Path to the `w.h5` file from which the wall shear stresses will be computed. For example, `results/${destination}/${meshname}/stationary/FacetNormal_th_none/Dirichlet_noslip/Newtonian/w.h5`