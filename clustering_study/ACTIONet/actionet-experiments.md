
### Notes
 - R version failed to compile locally (`Ubuntu 22.04.3 LTS` with `R 4.1.1`), - Python version seems broken with missing core functions. 
 - The `DockerFile` to create an image fails - presumably because it builds it from the current state rather than a stable version of hte github codebase. 
 - Other users report similar problems in the [issues section](https://github.com/shmohammadi86/ACTIONet/issues).
 - This pre-built [docker image](https://hub.docker.com/layers/actionet/actionet/mini/images/sha256-fc969b94bf28dd3b4fc5b0c099949060254978637d282889926dd216bff8c7ae?context=explore) (found through their documentation) with `R` version of `ACTIONet 2.1.9` worked for our use case.
 - Step 1. Start the the container and access the shell.
   ```bash
     docker pull actionet/actionet:mini
     docker run -it -v ~/Local/datasets/actionet/:/data actionet/actionet:mini /bin/bash
   ```
 - Step 2. Lauch `R` through the docker shell and run analysis scripts.
 - Code for `ACTIONet 2.1.9` is [here](https://github.com/shmohammadi86/ACTIONet/tree/b1c78ee7), and corresponds to this [documentation](https://compbio.mit.edu/ACTIONet/tutorials/).
 - Using `.h5ad` files created with `anndata` version greater than `0.8.0` led to errors with `ACTIONet::Anndata2ACE`. Here we used version `0.7.5` to generate the input files.


### Analysis

The script to generate ACTIONet results:

```R
library(ACTIONet)

base_path <- "/data/"
for fname in c("mouse_smartseq_counts_20240130", 
               "mouse_10x_GABAergic_isoCTX_counts_20240129", 
               "mouse_10x_Glutamatergic_isoCTX_counts_20240129") {
    data_file = paste(c(base_path,fname,".h5ad"),collapse="")
    result_file = paste(c(base_path,"ACTIONet_",fname,".h5ad"),collapse="")
    ace <- AnnData2ACE(data_file)
    names(assays(ace)) <- "counts"
    ace = normalize.ace(ace) # creats "logcounts" within assays(ace)
    assays(ace)[["counts"]] <- NULL
    ace = reduce.ace(ace) # uses "logcounts" within assays(ace)
    ace = run.ACTIONet(ace, 
                       min_cells_per_arch=10)
    ACE2AnnData(ace,
                file = result_file,
                main_assay=NULL) 
}
```

 - `reduce` or `run.ACTIONet` requires less memory than `AnnData2ACE`.
 - Our `.h5ad` file contains count data.
 - Clustering results with log1p-cpm data provided as part of the Anndata file and bypassing `normalize.ace` are worse.
 - Matrix $G$ referred to in the paper is part of the `ace` object: `colNets(ace)[[net_slot_out]] <- G` [(source)](https://github.com/shmohammadi86/ACTIONet/blob/R-release/R/main.R).
 - The default value for `net_slot_out` is `'ACTIONet'`.