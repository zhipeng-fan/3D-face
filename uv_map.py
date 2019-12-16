import scipy.io as sio
import numpy as np

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    # uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0])))) # add z
    return uv_coords

def gen_bilinear_map(uv_map):
    num_vertices, num_dim = uv_map.shape
    # [num_vertices, 2, 4]
    bilinear_uv_map = np.repeat(uv_map[:,:,None], 4, axis=-1)
    # [num_vertices, 4]
    bilinear_uv_weight_map = np.zeros((num_vertices, 4))
    # floor floor
    bilinear_uv_map[:,:,0] = np.floor(bilinear_uv_map[:,:,0])
    # floor ceil
    bilinear_uv_map[:,0,1] = np.floor(bilinear_uv_map[:,0,1])
    bilinear_uv_map[:,1,1] = np.ceil(bilinear_uv_map[:,1,1])
    # ceil floor
    bilinear_uv_map[:,0,2] = np.ceil(bilinear_uv_map[:,0,2])
    bilinear_uv_map[:,1,2] = np.floor(bilinear_uv_map[:,1,2])
    # ceil ceil
    bilinear_uv_map[:,:,3] = np.ceil(bilinear_uv_map[:,:,3])
    # Compute bilinear weights:
    rectangle_size_0 = np.abs(uv_map-bilinear_uv_map[:,:,0])
    rectangle_size_1 = np.abs(uv_map-bilinear_uv_map[:,:,1])
    rectangle_size_2 = np.abs(uv_map-bilinear_uv_map[:,:,2])
    rectangle_size_3 = np.abs(uv_map-bilinear_uv_map[:,:,3])

    bilinear_uv_weight_map[:,0] = rectangle_size_0[:,0]*rectangle_size_0[:,1]
    bilinear_uv_weight_map[:,1] = rectangle_size_1[:,0]*rectangle_size_1[:,1]
    bilinear_uv_weight_map[:,2] = rectangle_size_2[:,0]*rectangle_size_2[:,1]
    bilinear_uv_weight_map[:,3] = rectangle_size_3[:,0]*rectangle_size_3[:,1]
    
    assert (all((np.sum(bilinear_uv_weight_map, axis=1)-1)<1e-5))

    return bilinear_uv_map, bilinear_uv_weight_map

def load_uv_map(path="./BFM/BFM_UV.mat", img_size=224):
    """
    Load the uv mapping and prepared it for idxing in the deep 3DMM
    Return:
        Bilinear uv index map: [35709,2,4] (4 for grid style uv map)
        Bilinear uv weight: [35709, 4]
    Sample useage:
        predict uv map: uv_map = torch.randn(bs,3,224,224)
        uv_idx, uv_weight = load_uv_map()
        # sampled_value [bs, 3, 35709, 4]
        sampled_value = torch.stack([uv_map[:,:,uv_idx[:,0,0], uv_idx[:,1,0]],
                                     uv_map[:,:,uv_idx[:,1,1], uv_idx[:,1,1]],
                                     uv_map[:,:,uv_idx[:,1,2], uv_idx[:,1,2]],
                                     uv_map[:,:,uv_idx[:,1,3], uv_idx[:,1,3]],
                                    ], dim=-1)
        uv_weight = uv_weight[None, None, :, :]
        offset = torch.sum(sampled_value*uv_weightm, dim=-1).permute(0,2,1)
    """
    uv_mat = sio.loadmat(path)['UV']
    uv_mat = process_uv(uv_mat, img_size, img_size)
    uv_mat = np.array(uv_mat)
    
    index_exp = sio.loadmat('BFM/BFM_front_idx.mat')
    index_exp = np.array(index_exp['idx'].astype(np.int32) - 1) #starts from 0 (to 53215)

    cropped_uv_mat = uv_mat[index_exp[:,0]]

    bilinear_uv_map, bilinear_uv_weight = gen_bilinear_map(cropped_uv_mat)

    return bilinear_uv_map, bilinear_uv_weight



if __name__=="__main__":
    uv_map, uv_weight = load_uv_map()
    print (uv_map.shape, uv_weight.shape)