# Code and Data for AAAI'22 submission: GOCPT

- We propose a generalized online tensor factorization and completion framework, called ``GOCPT``. Our framework can support the most general tensor evolving scenarios with three typical types of evolving patterns. 
- Our GOCPT can unify the popular **online tensor factorization** and **streaming tensor completion** settings. Also, we can support other two settings: (i) **online tensor completion** (where tensor size does not change and the entries are gradually filled) and (ii) **factorization with changing values** (where the tensor size does not change and the entries can gradually updated).
- In this repo, we provide the code for (1) our GOCPT; (2) all baselines for all settings.

---
## 1. Package dependency

```bash
pip install scipy==1.5.0
pip install numpy==1.19.1, pickle
```
The dependency packages are pretty common. If any missing, please install that.

## 2. Quick start and reproducibility
Run our GOCPT and baselines on ``the most general case`` (default on the JHU Covid data)
```python
jupyter notebook general.ipynb
```

Run our GOCPT and baselines on ``online tensor factorization``
```python
cd ./src
python online_tensor_factorization.py --data synthetic
python online_tensor_factorization.py --data FACE
python online_tensor_factorization.py --data GCSS
```

Run our GOCPT and baselines on ``streaming tensor completion``
```python
cd ./src
python streaming_tensor_completion.py --data synthetic
python streaming_tensor_completion.py --data Indian
```

Run ablation study
```python
cd ./src
python ablation_study_on_sparsity.py
```

Run our GOCPT and baselines on ``online tensor completion``
```python
cd ./src
python online_tensor_completion.py --data synthetic
```

Run our GOCPT and baselines on ``factorization with changing values``
```python
cd ./src
python factorization_with_changing_value.py --data synthetic
```

- Note that, for each of the settings, our model can be easily applied to other tensor data by replacing the existing data.

## 3. Folder tree

- ./exp-data
    - FACE-3D.pkl: ORL Database of Faces contains 400 shots of face images with size 112 pixels by 99 pixels. The download link is here https://cam-orl.co.uk/facedatabase.html.
    - GCSS.pkl: This is a dataset, borrowed from https://github.com/ycq091044/MTC. This data contains google covid-19 symptom search results during the complete year 2020. The data can be viewed as a third-order tensor: 50 states, 422 keywords, and 362 days. The original link is here https://pair-code.github.io/covid19_symptom_dataset/ and we use the processing files provided by the above repo to generate the GCSS.pkl.
    - Indian_pines_corrected.mat: This is also an open dataset, containing 200 shots of hyperspectral images with size 145 pixels by 145 pixels. The link is here https://purr.purdue.edu/publications/1947/1.
    - jhu_covid_tensor.pickle: This is a dataset, borrowed from ``Qian, C., Kargas, N., Xiao, C., Glass, L., Sidiropoulos, N. and Sun, J., Multi-version Tensor Completion for Time-delayed Spatio-temporal Data. IJCAI 2021.`` The original dataset is open and collected from https://github.com/CSSEGISandData/COVID-19. 
- ./src
    - ablation_study_on_sparsity.py: the script is used to evaluate the performance of sparse or dense strategy proposed in our paper for Experimental Section 5.5.
    - online_tensor_factorization.py: this script is used to evaluate on one of the most popular ``online tensor factorization`` setting, used in experimental section 5.3 and appendix section B.3.
    - streaming_tensor_completion.py: this script is used to evaluate on another the most popular ``streaming tensor completion`` setting, used in experimental section 5.4 and appendix section B.4.
    - factorization_with_changing_value.py: this script is used for appendix section B.6.
    - online_tensor_completion.py: this script is used for appendix section B.5.
    - model.py: this is the util file that contains all common or iteration modules used in other scripts.
- general.ipynb: this is the scripts used in evaluating on the most general setting.

## 4. Synthetic data generation and evaluation
### 4.1 Generate the synthetic data
- In all settings, the synthetic data are generated the same way as follows
```python
# for example, if we want to generate a low-rank tensor 
#   with I1, I2, I3, R = (100, 100, 500, 5)
I, J, K, R = 100, 100, 500, 5
A0 = np.random.random((I, R))
B0 = np.random.random((J, R))
C0 = np.random.random((K, R))
X = np.einsum('ir,jr,kr->ijk',A0,B0,C0)
```

### 4.2 Evaluation on the synthetic data
- we have encoded the synthetic data generation code in each script
- directly call synthetic data, and the results will appear
```python
cd ./src
python online_tensor_factorization.py --data synthetic
python streaming_tensor_completion.py --data synthetic
python online_tensor_completion.py --data synthetic
python factorization_with_changing_value.py --data synthetic
python ablation_study_on_sparsity.py
```

### 4.3 Evaluation on real-world data
- it is trivial to replace the ``synthetic data generation module`` to ``[you own real data]`` and the rest of the files can run automatically.


## 5. Future work
- We plan to refactorize the code into torch version and support ``cuda`` to accelerate our computation
- We will consider more tensor evolving patterns and extend our frameworks, such as tensor rank can change or the mode can shrink during the evolution.
- As a long-term plan, we will release a python package called ``GOCPT``, which provides the fast/distributed/parallel implementations of the baselines and our GOCPT models for subsequent research.