# MMIDAS (Mixture Model Inference with Discrete-coupled AutoencoderS)

Implementation of [Joint inference of discrete cell types and continuous type-specific variability in single-cell datasets with MMIDAS](https://www.biorxiv.org/content/10.1101/2023.10.02.560574v1.abstract).

A generalized and unsupervised mixture variational model with a multi-armed deep neural network, to jointly infer the discrete type and continuous type-specific variability. This framework can be applied to analysis of both, uni-modal and multi-modal datasets. It outperforms comparable models in inferring interpretable discrete and continuous representations of cellular identity, and uncovers novel biological insights. MMIDAS can thus help researchers identify more robust cell types, study cell type-dependent continuous variability, interpret such latent factors in the feature domain, and study multi-modal datasets.

![](MMIDAS.png)
## Data
- [Allen Institute Mouse Smart-seq dataset](https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq)
- [Allen Institute Mouse 10x isocortex dataset](https://assets.nemoarchive.org/dat-jb2f34y)
- [Allen Institute Patch-seq data](https://dandiarchive.org/dandiset/000020/)

  The electrophysiological features have been computed following the approach in [cplAE_MET/preproc/data_proc_E.py](cplAE_MET/preproc/data_proc_E.py)
- [Seattle Alzheimer’s disease dataset (SEA-AD)](https://SEA-AD.org/)

## Environment
```
conda create -n mmidas python=3.7
conda install scipy scikit-learn jupyterlab pandas seaborn
pip install flatten-json feather-format tensorboard h5py hdf5plugin toml
# CUDA 11.2
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 pytorch-cuda=11.2 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.11.0 torchvision==0.13.1 torchaudio==0.11.0 cpuonly -c pytorch
```
