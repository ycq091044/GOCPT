import numpy as np
from .utils import rec_from_factors

# ------------ simulation scripts --------------------

def growth_1d(X, prep=0.5, inc=1):
    """
    Generate the simulation setting from a tensor with "time" as the last mode
    INPUT:
        - <tensor> X: (I1, I2, ..., In)
        - <int> prep: the percentage of preparation data
        - <int> inc: how many new slices at the next step
    OUTPUT:
        - <tensor> X_0: the prepration tensor
        - <tensor list> X_inc_ls: a list of new tensors that appear later
    """
    if type(X) != type([1,2]):
        In = X.shape[-1]
        prep_threshold = round(In * prep)
        X_0 = X[..., :prep_threshold]
        X_inc_ls = []
        for i in range(prep_threshold+1, In, inc):
            X_inc_ls.append(X[..., i:i+inc])

        message = """
        ---------- new OTF setting ------------
        base tensor size: {},
        new tensor increment size: {},
        tensor will be updated {} times.
        """.format(X_0.shape, X_inc_ls[0].shape, len(X_inc_ls))
        print (message)

        return X_0, X_inc_ls

    else:
        X, mask = X
        In = X.shape[-1]
        prep_threshold = round(In * prep)
        X_0 = X[..., :prep_threshold]
        mask_0 = mask[..., :prep_threshold]
        X_inc_ls, mask_inc_ls = [], []
        for i in range(prep_threshold+1, In, inc):
            X_inc_ls.append(X[..., i:i+inc])
            mask_inc_ls.append(mask[..., i:i+inc])

        message = """
        ---------- new OTC setting ------------
        base tensor size: {},
        new tensor increment size: {},
        tensor will be updated {} times.
        """.format(X_0.shape, X_inc_ls[0].shape, len(X_inc_ls))
        print (message)

        return [X_0, mask_0], [X_inc_ls, mask_inc_ls]


def value_update(X, mask, percent=0.1, amp=0.15):
    """
    to simulate the value update scenario
    INPUT:
        - <tensor> X: the masked tensor
        - <tensor> mask: the mask itself
        - <int> or <float>: percentage of changed elements or how many elements to change
        - <float> amp: the amplitude of uniform noise to the value
    OUTPUT:
        - <list> coords: coordinate list of the changed elements
        - <list> values: new value list of the changed elements
    """

    nonzero_coord = np.where(mask>0.5)
    nonzero_num = len(nonzero_coord[0])
    if percent < 1.0:
        # we think it means percentage change
        selected = np.random.choice(np.arange(nonzero_num), round(nonzero_num * percent), replace=False) 
    else:
        # we think it is absolute number change
        percent_num = round(percent)
        selected = np.random.choice(np.arange(nonzero_num), percent_num, replace=False) 

    # get the coordinates of value updates
    coords = [coord[selected] for coord in nonzero_coord]
    # get the original values in those coordinates
    values = X[coords] 
    values *= (1 + amp * np.random.random(values.shape))

    message = "number of newly updated entries: {} / {}".format(len(selected), nonzero_num)
    print (message)

    return coords, values


def missing_fill(X, mask, percent=0.1, factors=None):
    """
    to simulate the missing filling scenario
    INPUT:
        - <tensor> X: the masked tensor
        - <tensor> mask: the mask itself
        - <int> or <float>: percentage of changed elements or how many elements to fill
        - <matrix list> factors: it is not necessary. However, using factors during the simulation can \
            provide a smoothed missing filling.
    OUTPUT:
        - <list> coords: coordinate list of the changed elements
        - <list> values: new value list of the changed elements
    """

    zero_coord = np.where(mask<0.5)
    zero_num = len(zero_coord[0])
    if percent < 1.0:
        # we think it means percentage fill
        selected = np.random.choice(np.arange(zero_num), round(zero_num * percent), replace=False) 
    else:
        # we think it is absolute number fill
        percent_num = round(percent)
        selected = np.random.choice(np.arange(zero_num), percent_num, replace=False) 

    if factors is None:
        # then we sample some exsiting tensor elements
        coords = [coord[selected] for coord in zero_coord]
        values = np.random.choice(X[np.where(mask > 0.5)], len(selected), replace=True)
    else:
        # we sample some elements from the reconstruction
        coords = [coord[selected] for coord in zero_coord]
        reconstruction = rec_from_factors(factors)
        values = reconstruction[coords]

    message = """number of newly filled entries: {} / {}""".format(len(selected), zero_num)
    print (message)
    
    return coords, values
