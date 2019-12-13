import scipy.io as sio
import numpy as np

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0])))) # add z
    return uv_coords

def load_uv_map(path="./data/BFM_UV.mat", img_size=224):
    uv_mat = sio.loadmat(path)['UV']
    uv_mat = process_uv(uv_mat, img_size, img_size)
    print (uv_mat)

if __name__=="__main__":
    print (load_uv_map().keys())