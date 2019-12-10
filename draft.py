import BFM
import os
import trimesh
import numpy as np
import utils
import torch
from skimage import io

if not os.path.isfile("./BFM/BFM_model_front.mat"):
    utils.transferBFM09()

# bfm_face_model = BFM.BFM()
# print (bfm_face_model.point_buf[-1,-1])
# mesh = trimesh.Trimesh(vertices=bfm_face_model.meanshape.reshape(-1,3), 
#                        faces=bfm_face_model.tri.reshape(-1,3)-1, 
#                        vertex_colors=bfm_face_model.meantex.reshape(-1,3))

# mesh.show()

# def morph_face(bfm_model, id_param=None, ex_param=None, tex_param=None):
#     if id_param is None:
#         id_param = np.random.randn(80, 1)
#     if ex_param is None:
#         ex_param = np.random.randn(64, 1)
#     if tex_param is None:
#         tex_param = np.random.randn(80,1)

#     vertices = bfm_face_model.meanshape+np.dot(bfm_face_model.idBase, id_param)#+np.dot(bfm_face_model.exBase, ex_param)
#     colors = np.clip(bfm_face_model.meantex+np.matmul(bfm_face_model.texBase, tex_param), 0, 255)
#     mesh = trimesh.Trimesh(vertices=vertices.reshape(-1,3), 
#                             faces=bfm_face_model.tri.reshape(-1,3)-1, 
#                             vertex_colors=colors.reshape(-1,3))

#     mesh.show()


# morph_face(bfm_face_model)

# print (bfm_face_model.meanshape.shape)
# print (bfm_face_model.texBase.shape)
# print (bfm_face_model.exBase.shape)
# print (bfm_face_model.meantex.shape)
# print (bfm_face_model.idBase.shape)

# print (bfm_face_model.tri.shape)

import soft_renderer as sr

camera_distance = 2.732
elevation = 0
azimuth = 180

renderer = sr.SoftRenderer(image_size=512, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30,
                            perspective=True, light_intensity_directionals=0)

renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)

bs = 10

face_model = BFM.BFM_torch()
shape = face_model.get_shape(torch.randn(bs,80), torch.zeros(bs,64))
albedo = face_model.get_texture(torch.randn(bs,80))

shape = shape.reshape(bs,-1,3)
albedo = albedo.reshape(bs,-1,3)/255.

# face_id = face_model.tri-1
# face_id = face_id[None,:,:].expand(bs,-1,-1)

# print (shape[:, face_id.long(), :].shape)

# v1 = torch.gather(shape, dim=1, index=face_id[:,:,:1].expand(-1,-1,3).long())

# bs_id = 5
# face_idx = 10
# print (shape[bs_id, face_id[bs_id,face_idx,0].long(),:])
# print (v1[bs_id,face_idx,:])

# print (face_model.keypoints.shape)


# rot_param = torch.randn(bs,3)
# rot_mat = face_model.compute_rotation_matrix(rot_param)
# print (shape.shape)
# # shape = torch.bmm(shape, rot_mat)

# mesh = trimesh.Trimesh(vertices=shape[0].numpy(), 
#                        faces=face_model.tri.reshape(-1,3)-1, 
#                        vertex_colors=np.clip(albedo[0].numpy(),0,1))
# mesh.show()

mesh, image = renderer(shape.cuda(), (face_model.tri.reshape(-1,3)-1)[None,:,:].expand(bs,-1,-1).cuda(), albedo.cuda(), texture_type="vertex")
import pdb; pdb.set_trace()

# lmk = mesh.vertices[:, face_model.keypoints, :].cpu().numpy()
# print (lmk.shape)
# lmk[:,:,:2] = (lmk[:,:,:2]+1)*512/2
# lmk[:,:,1] = 512-lmk[:,:,1]
# print (lmk)

# io.imsave("./test_ori.png",image[0].permute(1,2,0).cpu().numpy())


# shape += torch.Tensor([0.5,0,0])

# mesh = trimesh.Trimesh(vertices=shape[0].numpy(), 
                    #    faces=face_model.tri.reshape(-1,3)-1, 
                    #    vertex_colors=np.clip(albedo[0].numpy(),0,1))
# mesh.show()

# image = renderer(shape[0].cuda(), (face_model.tri.reshape(-1,3)-1).cuda(), (albedo[0]).cuda(), texture_type="vertex")
# io.imsave("./test_shift.png",image[0].permute(1,2,0).cpu().numpy())

# image = renderer(shape.reshape(-1,3).cuda(), (face_model.tri.reshape(-1,3)-1).cuda(), (albedo.reshape(-1,3)/255.).cuda(), texture_type="vertex")

# print (image[0].permute(1,2,0).cpu().numpy().shape)

# io.imsave("./test.png",image[0].permute(1,2,0).cpu().numpy())

# from dataset import CACDDataset
# import matplotlib.pyplot as plt
# TrainSet = CACDDataset('./data/CACD2000_test.hdf5')
# img, lmk = TrainSet[1]
# plt.imshow(image[0].permute(1,2,0).cpu().numpy())
# plt.scatter(lmk[0,:,0], lmk[0,:,1])
# plt.show()