from PIL import Image
import numpy as np
import pickle

def read_img(path):
    im = Image.open(path)
    # print(im.size) # 92 x 112
    return np.array(im)

if __name__ == "__main__":

    img_collect = []
    for j in range(1, 11):
        for i in range(1, 41):
            path = './FACE/s{}/{}.pgm'.format(i, j)
            img_collect.append(read_img(path))
    result = np.array(img_collect)
    result2 = np.transpose(result, (1,2,0))
    pickle.dump(result2, open('./exp-data/FACE-3D.pkl', 'wb'))
    # print (result.shape)
    

