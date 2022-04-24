from os import pread
import sys
from xml.etree.ElementPath import prepare_descendant
import numpy as np

sys.path.insert(0, '/home/chaoqiy2/github/GOCPT')
from GOCPT import datasets, utils, otf, otc, gtf, gtc

def test_dataset():
    FACE_3D = datasets.FACE_3D()
    GCSS = datasets.GCSS()
    print ('tensor size: {}, {}'.format(GCSS.shape, FACE_3D.shape))

def test_syn_data():
    complete_tensor = datasets.syn_data(R=5, size=(5,10,15,20), dist='unif')
    print ('tensor size: {}'.format(complete_tensor.shape))
    [masked_tensor, mask] = datasets.syn_data(R=5, size=(15, 5, 8), \
        dist='normal', with_mask=0.95)
    print ('tensor/mask size: {}, {}'.format(complete_tensor.shape, \
        masked_tensor.shape, mask.shape))

def test_otf():
    # prepare the online tensor factorization setting
    X = datasets.syn_data(R=5, size=(5,15,100), dist='unif')
    [X_0, X_inc_ls] = utils.generate_simulation(X, prep=0.3, inc=1)

    # model = otf.OnlineCPD(X_0, R=5, iters=100)
    # model = otf.MAST(X_0, R=5, iters=100)
    # model = otf.CPStream(X_0, R=5, iters=100)
    model = otf.OnlineCPD(X_0, R=5, iters=100)

    for increments in X_inc_ls:
        model.update(increments, iters=3, verbose=True)

def test_otc():
    # prepare the online tensor factorization setting
    [masked_X, mask] = datasets.syn_data(R=5, size=(5,15,100), dist='unif', with_mask=0.9)
    [[X_0, mask_0], [X_inc_ls, mask_inc_ls]] = utils.generate_simulation([masked_X, mask], prep=0.3, inc=3)

    model = otc.OnlineSGD([X_0, mask_0], R=5, iters=100)
    # model = otc.OLSTEC([X_0, mask_0], R=5, iters=100)
    # model = otc.GOCPTE([X_0, mask_0], R=5, iters=100)
    # model = otc.GOCPT([X_0, mask_0], R=5, iters=100)

    for increments in zip(X_inc_ls, mask_inc_ls):
        model.update(increments, iters=3, verbose=True)

def test_gtf():
    # prepare the online tensor factorization setting
    X = datasets.syn_data(R=5, size=(5,15,100), dist='unif')
    [X_0, X_inc_ls] = utils.simulate_1d_growth(X, prep=0.3, inc=3)

    # model = gtf.GOCPTE(X_0, R=5, iters=100)
    model = gtf.GOCPTE(X_0, R=5, iters=100)

    for increments in X_inc_ls:
        if np.random.random() > 0.5:
            model.update(increments, iters=3, new_R=6)
        else:
            model.update(increments, iters=3, new_R=3)

def test_gtc():
    # prepare the online tensor factorization setting
    [X, mask] = datasets.syn_data(R=5, size=(5,5,15,100), dist='unif', with_mask=0.9)
    [X_0, mask_0], [X_inc_ls, mask_inc_ls] = utils.simulate_1d_growth([X, mask], prep=0.3, inc=3)

    # model = gtc.GOCPTE([X_0, mask_0], R=5, iters=100)
    model = gtc.GOCPTE([X_0, mask_0], R=5, iters=100)

    for X_inc, mask_inc in zip(X_inc_ls, mask_inc_ls):
        vupdate_coords, vupdate_values = utils.simulate_value_update(model.X, model.mask, percent=10, amp=0.05)
        fill_coords, fill_values = utils.simulate_miss_fill(model.X, model.mask, percent=10, factors=model.factors)
        if np.random.random() > 0.5:
            model.update([X_inc, mask_inc], iters=3, alpha=1e-1, value_update=[vupdate_coords, vupdate_values], \
                miss_fill=[fill_coords, fill_values], new_R=3)
        else:
            model.update([X_inc, mask_inc], iters=3, alpha=1e-1, value_update=[vupdate_coords, vupdate_values], \
                miss_fill=[fill_coords, fill_values], new_R=7)


test_gtc()