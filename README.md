# GOCPT
Generalized Online Canonical Polyadic Tensor Factorization and Completion

- This package ``GOCPT`` can support the following evolving patterns:
    - **mode growth** along one or more dimensions
    - **missing value filling** for incomplete tensors
    - **value updates** for previously incorrect inputs
    - **rank changes** during the evolution
- We provide two versions of our generalized model
    - ``GOCPT``: need to preserve all previous data and provides more accurate fit
    - ``GOCPTE (economy version)``: no previous data is needed and can provide fast speed
- Our GOCPT can unify the following common **online tensor evolution** settings

---
## 1. Package Installation
```bash
$ pip install GOCPT
```
To look up for help, directly type "GOCPT" in the cmd and the help message will pop up.
```bash
$ GOCPT
```
## 2. Examples of Different Settings
### 2.1 Tensor Loader 
``GOCPT.datasets``
- datasets.GCSS
- datasets.FACE_3D
- datasets.JHU_COVID

The tensor data is formatted as ``np.ndarry`` type (the current version is compatible with numpy only, we are building torch version to support CUDA). They can be loaded from external data or can be synthetically generated from our scripts with various distribution.
```python
# create data from external sources
from GOCPT import datasets
GCSS = datasets.GCSS()
FACE_3D = datasets.FACE_3D()
```

```python
# creating synthetic tensors
full_tensor = datasets.syn_data(R=5, size=(5,10,15,20), dist='unif')
[masked_tensor, mask] = datasets.syn_data(R=5, size=(15, 5, 8), \
        dist='normal', with_mask=0.95)
```
### 2.2. Online Tensor Factorization (OTF)
``GOCPT.otf``
- Support arbitrary order tensors
    - otf.OnlineCPD
    - otf.CPStream
    - otf.MAST
    - otf.GOCPTE
    - otf.GOCPT
- Support only 3-order tensors
    - otf.SDT
    - otf.RLST

This setting needs a base tensor ``X_0`` for initial factor estimation and a list of tensor increments ``X_inc_ls`` for each subsequent updates. CURRENTLY, we only support the setting, where only the last tensor mode is evolving! Using a 4-th order tensor as an example, ``X_0: (3, 10, 15, 50)`` and ``x_inc_ls[0]: (3, 10, 15, 2)``, ``x_inc_ls[1]: (3, 10, 15, 5)``, ... 
```python
# generate simulation OTF setting
from GOCPT import utils
[X_0, X_inc_ls] = utils.generate_simulation(full_tensor, prep=0.3, inc=3)
    
# prepare model and run online updates
from GOCPT import otf
model = otf.GOCPTE(X_0, R=5, iters=100)
for increments in X_inc_ls:
    model.update(increments, verbose=True)
```
### 2.3. Online Tensor Completion (OTC)
``GOCPT.otc``
- otc.OLSTEC
- otc.OnlineSGD
- otc.GOCPTE
- otc.GOCPT

The settings are similar as OTF, while the base tensor and the increments are incomplete tensors with a mask.
```python
# generate simulation OTC setting
[[X_0, mask_0], [X_inc_ls, mask_inc_ls]] = \
        utils.generate_simulation([masked_X, mask], prep=0.3, inc=3)
    
# prepare model and run online updates
from GOCPT import otc
model = otc.GOCPTE([X_0, mask_0], R=5, iters=100)
for increments in zip(X_inc_ls, mask_inc_ls):
        model.update(increments, verbose=True)
```

### 2.4. Online Tensor Completion (OTC)
```python
# generate simulation OTC setting
[[X_0, mask_0], [X_inc_ls, mask_inc_ls]] = \
        utils.generate_simulation([masked_X, mask], prep=0.3, inc=3)
    
# prepare model and run online updates
from GOCPT import otc
model = otc.GOCPTE([X_0, mask_0], R=5, iters=100)
for increments in zip(X_inc_ls, mask_inc_ls):
        model.update(increments, verbose=True)
```


## 5. Future work
- We plan to refactorize the code into torch version and support ``cuda`` to accelerate our computation
- We will consider more tensor evolving patterns to extend the coverage of our package.
- As a long-term plan, we plan to support sparse tensor implementations for subsequent research.