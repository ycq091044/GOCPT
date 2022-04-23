from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import sys

from pyrsistent import inc

sys.path.insert(0, '/home/chaoqiy2/github/GOCPT')
from GOCPT import datasets, utils, otf, otc

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
    # model = otf.GOCPTE(X_0, R=5, iters=100)
    # model = otf.MAST(X_0, R=5, iters=100)
    # model = otf.CPStream(X_0, R=5, iters=100)
    # model = otf.OnlineCPD(X_0, R=5, iters=100)
    model = otf.GOCPT(X_0, R=5, iters=100)

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


test_otf()
