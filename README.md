#  Numerical evaluation of wall shear stress using FEM

# Prerequisities

1. **Legacy FEniCS 2019.1.0**  
   Installation guide can be found at [FEniCS Project Archive](https://fenicsproject.org/download/archive/).

2. **`ns_aneurysm` Package**  
   The package can be found on our [GitLab repository](https://gitlab.karlin.mff.cuni.cz/bio/aneurysm). To install it, use the following command:
   ```bash
   python3 -m pip install -e .
    ```

# 2D Stokes example

Results for the 2D Stokes flow on a unit square can be reproduced by running:
```bash
python3 python stokes_example.py -N 8 16 32 64 128 256 512 -g 0.01 --lambda 10 -o res_stokes_p1p1/ --element p1p1
```
for the P1/P1 stabilized element, and
```bash
python3 python stokes_example.py -N 8 16 32 64 128 256 512 -o res_stokes_th/ --element th
```
for the P2/P1 element.


# 3D Poiseuille flow

TODO

# 3D Patient-specific simulations

TODO