# KOMINTERN

The MOGP coefficients are stored in a pickle or json+zip file. File tree is :

```python
    myKomiFile.komi
        |- case (str)
        |- x_labels [str]
        |- x [[float32]]
        |- y_labels [str]
        |- y_means [float32]
        |- y_maxes [float32]
        |- x_bounds [[param1_min, param1_max], ... , [paramN_min, paramN_max]] # link to x_labels order
        |- kernel (str)
        |- "proj"
        |     |- lscales
        |     |- mean_cache
        |     |- Q
        |     |- R
        |     |- sigma_proj
        |     |- sigma_orth
        |- "var"
        |     |- lscales
        |     |- mean_cache
        |     |- lmc_coeff
        |     |- inducing_points
        |     |- distrib_covar
        |     |- noises
        |- "icm"
              |- lscales
              |- mean_cache
              |- lmc_coeff
              |- noises
```
