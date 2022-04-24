# GOCPT: Generalized Online CP Tensor Learning
Real-world tensor (multi-dimensional data) can evolve in various ways (as shown below). We provide this ``GOCPT`` package to handle the most general online Canonical Polyadic (CP) tensor factorization and completion problem.

<img src="https://github.com/ycq091044/GOCPT/raw/main/illustration/generalized_online_tensor_evolution.png" width="620">

- This package ``GOCPT`` can **unify most of the existing tensor factorization and completion scenarios** and can support the following evolving patterns in the online setting:
    - **mode growth** along one or more dimensions
    - **missing value filling** for incomplete tensors
    - **value updates** for previously incorrect inputs
    - **rank changes** during the evolution
- This package provides two model for handling generalized online tensor factorization or completion problems. These two models have **comparable or better accuracy/fitness/complexity than previous baselines**. 
    - ``GOCPT``: need to preserve all previous data and provides more accurate fit
    - ``GOCPTE (economy version)``: no previous data is needed and can provide fast speed

---
## 1. Package Installation
```bash
$ pip install GOCPT
```
To look up for help, directly type "GOCPT" in the cmd and the help message will pop up.
```bash
$ GOCPT
```
We provide the following modules:
- ``GOCPT.datasets``: external real tensor loader, synthetic tensor generation
- ``GOCPT.otf``: baseline models for common online tensor factorization
- ``GOCPT.otc``: baseline models for common online tensor completion
- ``GOCPT.gotf``: our models for generalized online tensor factorization
- ``GOCPT.gotc``: our models for generalized online tensor completion
- ``GOCPT.simulation``: our simulation environments for all online tensor settings

## 2. Generalized Online Tensor Factorization (GOTF)
The GOTF setting is a generalization of online tensor factorization (OTF) setting. Here, multiple evolution patterns can appear simultaneously:
- **mode growth**: one or more tensor modes can grow 
- **rank change**: the rank of the maintained factors can increase or decrease

We provide the following two models for the GOTF setting:
- ``gotf.GOCPTE``: no previous data is needed and can provide fast speed
- ``gotf.GOCPT``: need to preserve all previous data and provides more accurate fit

and the following models for the common OTF setting:
- Support arbitrary order tensors: ``otf.OnlineCPD``, ``otf.CPStream``, ``otf.MAST``
- Support only 3-order tensors and 1 new slice at each step: ``otf.SDT``, ``otf.RLST``
### A Quick Example
```python
from GOCPT import datasets, simulation, otf, gotf
import numpy as np

# load Columbia FACE 3D tensor
X = datasets.FACE_3D()

# simulate one mode growth (typically the last mode)
# use 30% as preparation and increase 3 slices at each step
[X_0, X_inc_ls] = simulation.growth_1d(X, prep=0.3, inc=3)

# initialize the model
model = gotf.GOCPTE(X_0, R=5, iters=100)

# update model with new slices, new_R can change during for-loop
for increments in X_inc_ls[:10]:
    new_R = np.random.choice([5,6,7,8])
    model.update(increments, new_R=new_R)
```
## 3. Generalized Online Tensor Completion (GOTC)
The GOTC setting is a generalization of online tensor completion (OTC) setting. Here, multiple evolution patterns can appear simultaneously:
- **mode growth**: one or more tensor modes can grow 
- **value update**: previously observed entries may change due to new information
- **missing filling**: some previous missing values may be received (delayed feedback)
- **rank change**: the rank of the maintained factors can increase or decrease

We provide the following two models for the GOTC setting:
- ``gotc.GOCPTE``: no previous data is needed and can provide fast speed
- ``gotc.GOCPT``: need to preserve all previous data and provides more accurate fit

and the following models for the OTC setting: ``otc.OLSTEC``, ``otc.OnlineSGD``
### A Quick Example
```python
from GOCPT import datasets, simulation, otc, gotc
import numpy as np

# load synthetic data
[masked_X, mask] = datasets.syn_data(R=5, size=(5, 10, 15, 100), dist='unif', sparsity=0.95)

# simulate one mode growth (typically the last mode)
# use 30% as preparation and increase 3 slices at each step
[[X_0, mask_0], [X_inc_ls, mask_inc_ls]] = simulation.growth_1d([masked_X, mask], prep=0.3, inc=7)

# initialize the model
model = gotc.GOCPTE([X_0, mask_0], R=5, iters=100)

# update model with new slices, new_R can change during for-loop
for increments in zip(X_inc_ls, mask_inc_ls):
    # simulate value update， missing filling，change of rank
    new_value_update = simulation.value_update(model.X, model.mask, percent=10, amp=0.05)
    new_missing_fill = simulation.missing_fill(model.X, model.mask, percent=10, \
        factors=model.factors)
    new_R = np.random.choice([5,6,7,8])
    model.update(increments, new_R=new_R, value_update=new_value_update, \
        miss_fill=new_missing_fill)
```

## 4. How to use the modules?
### 4.1. Data Loader ``GOCPT.datasets``
- ``datasets.GCSS``
- ``datasets.FACE_3D``

The tensor data is formatted as ``np.ndarry`` type (the current version is compatible with numpy only, we are building torch version to support CUDA). They can be loaded from external data or can be synthetically generated from our scripts with various distribution.
```python
# create data from external sources
from GOCPT import datasets
GCSS = datasets.GCSS()
FACE_3D = datasets.FACE_3D()
```

```python
"""
INPUT:
    - <int> R: tensor rank
    - <list/tuple> size: tensor mode specification, e.g., [5,10,15]
    - <str> dist: in which distribution
    - <float> sparsity: sparsity of the tensor, default is None
OUTPUT:
    if sparsity is not None:
        - <tensor> masked_X
        - <tensor> mask
    if sparsity is None:
        - <tensor> full_tensor
"""
# creating full synthetic tensors
full_tensor = datasets.syn_data(R=5, size=(5,10,15,20), dist='unif')

# creating incomplete synthetic tensors with mask
[masked_tensor, mask] = datasets.syn_data(R=5, size=(15, 5, 8), dist='normal', with_mask=0.95)
```
### 4.2. Simulation ``GOCPT.simulation``
For real world applications, the **base tensor** (sometimes with **tensor mask**), **new tensor slices per time step**, **change of rank**, **new value update**, **new missing fillings** will be given. In order to simulating all different settings, we provide the following functions
- ``simulation.growth_1d``: currently, we only support one mode growth
- ``simulation.value_update``: simulate value updates in previous entries
- ``simulation.missing_fill``: simulate values to fill previous missing entries

```python
"""
INPUT:
    - <tensor> X or [<tensor>, <tensor>] masked_X, mask: for factorization or completion
    - <int> prep: the percentage of preparation data (along the last mode)
    - <int> inc: how many new slices at the next step (along the last mode)
OUTPUT:
    - <tensor> X_0 or [<tensor>, <tensor>] X0, mask0: the prepration tensor
    - <tensor list> X_inc_ls or [<tensor list>, <tensor list>] X_inc_ls, mask_inc_ls : a list \
    of new tensors that appear later
"""
# simulate mode growth for factorization
[X_0, X_inc_ls] = simulation.growth_1d(X, prep=0.3, inc=3)
# simulate mode growth for completion
[[X_0, mask_0], [X_inc_ls, mask_inc_ls]] = simulation.growth_1d([masked_X, mask], prep=0.3, inc=3)
```

```python
"""
INPUT:
    - <tensor> X: the masked tensor
    - <tensor> mask: the mask itself
    - <int> or <float>: percentage of changed elements or how many elements to change
    - <float> amp: the amplitude of uniform noise to the value
OUTPUT:
    - <list> coords: coordinate list of the changed elements
    - <list> values: new value list of the changed elements
"""
new_value_update = simulation.value_update(model.X, model.mask, percent=10, amp=0.05)
```
```python
"""
INPUT:
    - <tensor> X: the masked tensor
    - <tensor> mask: the mask itself
    - <int> or <float>: percentage of changed elements or how many elements to fill
    - <matrix list> factors: it is not necessary. However, using factors during the \
        simulation can provide a smoothed missing filling. If factors is None, then \
        we random sample existing elements
OUTPUT:
    - <list> coords: coordinate list of the changed elements
    - <list> values: new value list of the changed elements
"""
new_missing_fill = simulation.missing_fill(model.X, model.mask, percent=10, factors=model.factors)
```
### 4.3. Factorization Model Config ``GOCPT.gotf``, ``GOCPT.otf``
All models in this package will be fed on the initial tensor and store an intial list of factors. Then, during the evolution of the tensor, the model (a list of low-rank factors) is updated based on the all the accessible new information (from up to two different evolutions). For calculating the reconstruction error (percentage of fitness, PoF), we will still store all the information in the model class, though they will only be used in ``gotf.GOCPT``.
- the stats of initialization and each updates will be summarized after optimization
```python
# model initialization
"""
INPUT:
    - <tensor> X_0: the initial tensor
    - <int> R: tensor rank
    - <int> iters: [optinal default=50] for initial cpd-als iterations 
"""
model = gotf.GOCPT(X_0, R=5)
model = gotf.GOCPTE(X_0, R=5, alpha=1) # a special **alpha** is for weighting the previous results
baseline = otf.BASELINE(X_0, R=5)

# model update
"""
INPUT:
    - <tensor> increments: new tensor slices (along the last mode)
    - <int> new_R: [optional] new tensor rank change, only work for GOCPT.gotf model
"""
model.update(increments, new_R=new_R)
baseline.update(increments)
```
### 4.4. Completion Model Config ``GOCPT.gotc``, ``GOCPT.otc``
All models in this package will be fed on the initial tensor and the mask and store an intial list of factors. Then, during the evolution of the tensor, the model (a list of low-rank factors) is updated based on the all the accessible new information (from up to four different evolutions). For calculating the reconstruction error (percentage of fitness, PoF), we will still store all the information in the model class, though they will only be used in ``gotc.GOCPT``.
- the stats of initialization and each updates will be summarized after optimization
```python
# model initialization
"""
INPUT:
    - [<tensor>, <tensor>] X_0, mask_0: the initial tensor and initial mask
    - <int> R: tensor rank
    - <int> iters: [optinal default=50] for initial cpc-als iterations 
"""
model1 = gotc.GOCPT([X_0, mask_0], R=5)
model = gotf.GOCPTE([X_0, mask_0], R=5, alpha=1) # a spefical **alpha** here is for weighting previous results
baseline = otc.BASELINE([X_0, mask_0], R=5)

# model update
"""
INPUT:
    - [<tensor>, <tensor>] X_increments, mask_increments: new tensor and mask slices (along the last mode)
    - <int> new_R: [optional] new tensor rank change, only work for GOCPT.gotf model
"""
**kwargs = {
    "new_R": new_R, # new rank change
    "value_update": [value_update_coordinates, value_update_values], # new value updates
    "missing_fill": [missing_fill_coordinates, missing_fill_values]  # new missing fill
}
model.update([X_increments, mask_increments], **kwargs)
baseline.update([X_increments, mask_increments])
```


## 5. Future work
- We plan to refactorize the code into torch version and support ``cuda`` to accelerate our computation
- We will consider more tensor evolving patterns to extend the coverage of our package.
- As a long-term plan, we plan to support sparse tensor implementations for subsequent research.

<!-- ### Citation
```bibtex
@inproceedings{yang2021safedrug,
    title = {GOCPT: Generalized Online Canonical Polyadic Tensor Factorization and Completion},
    author = {Yang, Chaoqi and Qian, Cheng and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022},
    year = {2022}
}
``` -->